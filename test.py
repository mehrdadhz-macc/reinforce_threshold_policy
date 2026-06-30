"""
Evaluate a saved B&P REINFORCE threshold policy on test data.

Runs greedy (deterministic) actions — no gradient, no parameter updates.
Reports per-day total reward and summary statistics.

Usage:
    venv/bin/python3 test.py --model outputs/models/policy_multihour.pt
    venv/bin/python3 test.py --model outputs/models/policy_multihour.pt --days 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import build_day_index, day_auction_mids, load_all
from src.environment import MultiHourMarketEnv
from src.threshold_policy import AlphaPolicy, compute_regimes


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path} …")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    rt_eff   = checkpoint.get("round_trip_eff", 1.0)
    n_levels = checkpoint.get("n_levels", 1)
    policy   = AlphaPolicy(round_trip_eff=rt_eff, n_levels=n_levels)
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()

    print("Trained for", checkpoint.get("days_trained", "?"), "days  |  lr =", checkpoint.get("lr", "?"))
    print("Policy parameters:")
    for k, v in checkpoint.get("param_snapshot", {}).items():
        print(f"  {k:<12} = {v:+.4f}")

    print("\nLoading test data …")
    cim, auc = load_all(split="test")

    day_index = build_day_index(cim, auc)
    n_days    = min(args.days, len(day_index)) if args.days else len(day_index)
    print(f"Days available: {len(day_index)}  |  evaluating on {n_days}\n")

    env = MultiHourMarketEnv(
        capacity_mwh             = 200.0,  # paper §VI: 200 MWh storage unit
        max_charge_mw            = 50.0,   # paper §VI: 50 MW
        max_discharge_mw         = 50.0,
        efficiency               = 0.95,
        initial_soc_mwh          = 0.0,
        terminal_penalty_eur_mwh = 0.0,   # paper §V-C: no residual value at end of day
    )

    print(f"{'ep':>5}  {'date':<12}  {'reward':>12}  {'steps':>7}")
    print("─" * 48)

    rewards: list[float] = []

    with torch.no_grad():
        for ep_idx in range(n_days):
            day, delivery_starts, session_start = day_index[ep_idx]

            day_cim = cim[cim["delivery_start"].isin(delivery_starts)]
            day_auc = auc[auc["delivery_start"].isin(delivery_starts)]

            auction_mids = day_auction_mids(day_auc, delivery_starts)
            if auction_mids is None:
                print(f"{ep_idx+1:5d}  {str(day.date()):<12}  skipped (missing auction data)")
                continue
            if auction_mids.max() - auction_mids.min() <= 0:
                print(f"{ep_idx+1:5d}  {str(day.date()):<12}  skipped (flat auction curve)")
                continue

            regime = compute_regimes(auction_mids)
            policy.set_regime_info(regime)
            policy.set_env_params(
                env.capacity, env.initial_soc, env.max_charge, env.max_discharge
            )

            state        = env.reset(day_cim, delivery_starts, session_start)
            total_reward = 0.0
            n_steps      = 0
            done         = False

            while not done:
                actions              = policy.greedy_batch(
                    state.active_hours,
                    state.best_bid,
                    state.best_ask,
                    state.order_book,
                    state.elapsed_hours,
                    state.v_end_fraction,
                    position=state.position,
                    step_hours=state.step_hours,
                )
                state, reward, done, _ = env.step(actions)
                total_reward += reward
                n_steps      += 1

            rewards.append(total_reward)
            print(f"{ep_idx+1:5d}  {str(day.date()):<12}  {total_reward:12.2f}  {n_steps:7d}")

    if rewards:
        arr = np.array(rewards)
        print("\n" + "─" * 48)
        print(f"  Days evaluated : {len(arr)}")
        print(f"  Mean reward    : {arr.mean():12.2f} EUR")
        print(f"  Std reward     : {arr.std():12.2f} EUR")
        print(f"  Min reward     : {arr.min():12.2f} EUR")
        print(f"  Max reward     : {arr.max():12.2f} EUR")
        print(f"  Total reward   : {arr.sum():12.2f} EUR")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate B&P multi-hour REINFORCE policy on test data"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to saved policy checkpoint (.pt file)"
    )
    parser.add_argument(
        "--days", type=int, default=None,
        help="Number of test days to evaluate (default: all available)"
    )
    evaluate(parser.parse_args())
