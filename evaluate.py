"""
Evaluate a trained REINFORCE threshold policy on the test set.

Runs the greedy policy and the RI-LP benchmark on every test day, collects
per-day and per-tick metrics, then generates six diagnostic figures.

Usage:
    venv/bin/python3 evaluate.py --model outputs/models/policy_multihour.pt
    venv/bin/python3 evaluate.py --model outputs/models/policy_multihour.pt \\
                                 --plot-dir outputs/eval_plots --stride 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_all
from src.environment import MultiHourMarketEnv, N_HOURS
from src.reinforce_trainer import REINFORCEAgent
from src.ri_benchmark import run_ri_benchmark
from src.eval_plots import (
    DayResult, ThresholdData, TickRecord, save_all_plots,
)
from train import _build_day_index, _day_auction_mids


# ── Episode runner ─────────────────────────────────────────────────────────────

def _run_policy_episode(
    env           : MultiHourMarketEnv,
    agent         : REINFORCEAgent,
    day_cim       : "pd.DataFrame",
    delivery_starts: list,
    session_start  : "pd.Timestamp",
    tick_stride    : int,
    record_ticks   : bool,
    threshold_data : ThresholdData,
    fast           : bool = True,
    td_sample_every: int = 5,
) -> tuple[float, np.ndarray, np.ndarray, list[TickRecord] | None]:
    """
    Run one greedy-policy episode and collect metrics.

    Returns:
        (total_reward, final_soc, revenue_per_hour, tick_records)
        tick_records is None when record_ticks=False.
    """
    state          = env.reset(day_cim, delivery_starts, session_start, tick_stride=tick_stride)
    revenue_per_hour = np.zeros(N_HOURS)
    tick_records: list[TickRecord] = [] if record_ticks else None
    tick_idx       = 0
    total_reward   = 0.0
    done           = False

    while not done:
        active = state.active_hours

        # Compute threshold means — zero LP overhead (position=None path in policy)
        if active and (record_ticks or tick_idx % td_sample_every == 0):
            means = agent.policy.threshold_means_batch(
                active,
                state.best_bid,
                state.best_ask,
                state.elapsed_hours,
                state.v_end_fraction,
            )
        else:
            means = {}

        # Collect per-hour threshold vs price data for Plot 5 (all days, sampled)
        if means and tick_idx % td_sample_every == 0:
            for h in active:
                if h in means and h in state.best_bid and h in state.best_ask:
                    threshold_data.mu_Y.append(means[h]["mu_Y"])
                    threshold_data.bid.append(state.best_bid[h])
                    threshold_data.mu_X.append(means[h]["mu_X"])
                    threshold_data.ask.append(state.best_ask[h])
                    threshold_data.hour.append(h)

        # Example-day tick records for Plot 3
        if record_ticks and active and means:
            pos    = state.position
            t_sell = float(np.sum(np.maximum(0.0, pos)))
            t_buy  = float(np.sum(np.maximum(0.0, -pos)))
            m_bid  = float(np.mean([state.best_bid[h] for h in active]))
            m_ask  = float(np.mean([state.best_ask[h] for h in active]))
            m_muY  = float(np.mean([means[h]["mu_Y"] for h in active if h in means]))
            m_muX  = float(np.mean([means[h]["mu_X"] for h in active if h in means]))
            tick_records.append(TickRecord(
                elapsed_hours  = state.elapsed_hours,
                n_active       = len(active),
                mean_bid       = m_bid,
                mean_ask       = m_ask,
                mean_mu_Y      = m_muY,
                mean_mu_X      = m_muX,
                total_net_sell = t_sell,
                total_net_buy  = t_buy,
            ))

        # Greedy action and environment step (fast=True skips per-tick LP)
        actions                   = agent.greedy_actions(state, fast=fast)
        state, reward, done, info = env.step(actions)
        total_reward += reward

        for h, h_info in info.items():
            revenue_per_hour[h] += h_info.get("revenue", 0.0)

        tick_idx += 1

    final_soc = env.implied_soc()
    return total_reward, final_soc, revenue_per_hour, tick_records


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt = torch.load(args.model, map_location="cpu")
    print(f"Loaded model: {args.model}")
    print(f"  trained days : {ckpt.get('days_trained', '?')}")
    print(f"  round_trip_eff: {ckpt.get('round_trip_eff', 1.0)}")

    efficiency  = float(ckpt.get("round_trip_eff", 1.0))
    n_levels    = int(ckpt.get("n_levels", 1))
    hp_init     = ckpt.get("hp_init", {})

    agent = REINFORCEAgent(
        lr             = 1e-3,
        round_trip_eff = efficiency,
        n_levels       = n_levels,
        **hp_init,
    )
    agent.policy.load_state_dict(ckpt["state_dict"])
    agent.policy.eval()

    # ── Load test data ─────────────────────────────────────────────────────────
    print(f"\nLoading {args.split} data …")
    cim, auc = load_all(split=args.split)

    day_index = _build_day_index(cim, auc)
    n_days    = min(args.days, len(day_index)) if args.days else len(day_index)
    day_index = day_index[:n_days]
    print(f"Test days: {n_days}")

    # ── Environment ────────────────────────────────────────────────────────────
    env = MultiHourMarketEnv(
        capacity_mwh             = 200.0,
        max_charge_mw            = 50.0,
        max_discharge_mw         = 50.0,
        efficiency               = efficiency,
        initial_soc_mwh          = 0.0,
        terminal_penalty_eur_mwh = 0.0,
    )

    # Pick 3 example days: first, middle, last
    example_idxs = {0, n_days // 2, max(0, n_days - 1)}

    # ── Per-day stride: match training tick spacing ────────────────────────────
    unique_ts  = sorted(cim["timestamp"].unique())
    tick_secs  = max(1, round((unique_ts[1] - unique_ts[0]).total_seconds())) if len(unique_ts) >= 2 else 60
    stride     = args.stride

    results: list[DayResult] = []
    td = ThresholdData()

    header = f"{'day':>4}  {'policy':>12}"
    if args.with_ri:
        header += f"  {'RI-LP':>12}  {'ratio':>7}"
    print(f"\n{header}")
    print("─" * len(header))

    for day_idx, (day, delivery_starts, session_start) in enumerate(day_index):
        day_cim = cim[cim["delivery_start"].isin(delivery_starts)]
        day_auc = auc[auc["delivery_start"].isin(delivery_starts)]

        auction_mids = _day_auction_mids(day_auc, delivery_starts)
        if auction_mids is None or auction_mids.max() - auction_mids.min() <= 0:
            continue

        # RI-LP benchmark (opt-in — expensive: one LP solve per tick)
        if args.with_ri:
            ri_reward = run_ri_benchmark(
                env, day_cim, delivery_starts, session_start, tick_stride=stride
            )
        else:
            ri_reward = float("nan")

        # Policy — set regime info before episode
        agent.set_episode(
            auction_mids,
            capacity    = env.capacity,
            initial_soc = env.initial_soc,
            max_charge  = env.max_charge,
            max_dis     = env.max_discharge,
        )

        record = day_idx in example_idxs
        policy_reward, final_soc, rev_per_hour, tick_recs = _run_policy_episode(
            env, agent, day_cim, delivery_starts, session_start,
            tick_stride   = stride,
            record_ticks  = record,
            threshold_data= td,
            fast          = not args.with_lp_ri,
        )

        if args.with_ri:
            ratio = (policy_reward / ri_reward) if abs(ri_reward) > 1e-3 else float("nan")
            print(f"{day_idx:4d}  {policy_reward:12.2f}  {ri_reward:12.2f}  {ratio:7.3f}")
        else:
            print(f"{day_idx:4d}  {policy_reward:12.2f}")

        results.append(DayResult(
            date            = day,
            idx             = day_idx,
            policy_reward   = policy_reward,
            ri_reward       = ri_reward,
            final_soc       = final_soc,
            revenue_per_hour= rev_per_hour,
            tick_records    = tick_recs,
        ))

    if not results:
        print("No valid test days found.")
        return

    policy_mean = np.mean([r.policy_reward for r in results])
    print(f"\nMean policy reward : {policy_mean:.2f} EUR")
    if args.with_ri:
        ri_vals = [r.ri_reward for r in results if not np.isnan(r.ri_reward)]
        if ri_vals:
            ri_mean = np.mean(ri_vals)
            print(f"Mean RI-LP reward  : {ri_mean:.2f} EUR")
            if abs(ri_mean) > 1e-3:
                print(f"Policy / RI-LP     : {policy_mean / ri_mean:.3f}")

    # ── Generate plots ─────────────────────────────────────────────────────────
    save_all_plots(results, td, out_dir=args.plot_dir)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained threshold policy")
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "train"],
        help="Data split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to saved model checkpoint (.pt)"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Max number of test days to evaluate (default: all)"
    )
    parser.add_argument(
        "--stride", type=int, default=60,
        help="Tick stride for evaluation (default: 60 = hourly, ~24 ticks/day). "
             "Use 1 for finest resolution (slow)."
    )
    parser.add_argument(
        "--with-ri", action="store_true", dest="with_ri",
        help="Also run RI-LP benchmark for comparison (expensive: one LP per tick)."
    )
    parser.add_argument(
        "--with-lp-ri", action="store_true", dest="with_lp_ri",
        help="Use full LP-based RI reference inside the policy (matches training exactly, "
             "but slow: one LP per tick). Default: fast auction-mid approximation."
    )
    parser.add_argument(
        "--plot-dir", type=str, default="outputs/eval_plots", dest="plot_dir",
        help="Directory for output plots (default: outputs/eval_plots)"
    )
    evaluate(parser.parse_args())
