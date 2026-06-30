"""
Train the B&P REINFORCE threshold policy — multi-hour episode version.

One episode = one Berlin calendar day (all 24 delivery hours traded jointly).
The agent sees order books for all active delivery hours at each minute tick
and trades them independently using per-hour threshold sampling.

Usage:
    venv/bin/python3 train.py [--days N] [--lr LR] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.data_loader import load_all
from src.environment import MultiHourMarketEnv
from src.reinforce_trainer import REINFORCEAgent
from src.training_logger import TrainingLogger


# ── Hyperparameter sampling ───────────────────────────────────────────────────

# Ranges chosen from the policy structure (§V) and the appendix toy example.
_HP_RANGES: dict[str, tuple[float, float]] = {
    "alpha_s1_init"  : (0.0, 1.0),   # Step 1 anchor fraction within price range
    "alpha_b1_init"  : (0.0, 1.0),
    "alpha_s2_init"  : (0.0, 5.0),   # Step 2 SoC-pressure (EUR/MWh per unit v_end)
    "alpha_b2_init"  : (0.0, 5.0),
    "alpha_s3_init"  : (0.0, 2.0),   # Step 3 urgency magnitude (multiplier on range/2)
    "alpha_b3_init"  : (0.0, 2.0),
    "alpha_s4_init"  : (0.0, 2.0),   # Step 3 urgency rate (exp decay coefficient)
    "alpha_b4_init"  : (0.0, 2.0),
    "alpha_s5_init"  : (0.0, 5.0),   # Step 4 RI auxiliary shift (EUR/MWh)
    "alpha_b5_init"  : (0.0, 5.0),
    "log_sigma_init" : (-0.5, 2.5),  # log σ → σ ∈ [0.6, 12] EUR/MWh
}

_HP_DEFAULTS: dict[str, float] = {
    "alpha_s1_init"  : 0.5,
    "alpha_b1_init"  : 0.5,
    "alpha_s2_init"  : 0.0,
    "alpha_b2_init"  : 0.0,
    "alpha_s3_init"  : 0.0,
    "alpha_b3_init"  : 0.0,
    "alpha_s4_init"  : 1.0,
    "alpha_b4_init"  : 1.0,
    "alpha_s5_init"  : 0.0,
    "alpha_b5_init"  : 0.0,
    "log_sigma_init" : 0.3,   # σ ≈ 1.35 EUR/MWh (close to appendix toy example)
}


def _sample_hyperparams(seed: int) -> dict[str, float]:
    """Draw all policy hyperparameters uniformly from their natural ranges."""
    rng = np.random.default_rng(seed)
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in _HP_RANGES.items()}


# ── Auction feature helpers ───────────────────────────────────────────────────

def _day_auction_mids(
    auc            : pd.DataFrame,
    delivery_starts: list[pd.Timestamp],
) -> np.ndarray | None:
    """
    Compute per-hour D-1 auction MID = (best_ask + best_bid) / 2 from the
    full auction order book.  Returns shape-(24,) array or None if any hour
    is missing.
    """
    sell_best = (
        auc[auc["side"] == "sell"]
        .groupby("delivery_start")["price_eur_mwh"]
        .min()
    )
    buy_best = (
        auc[auc["side"] == "buy"]
        .groupby("delivery_start")["price_eur_mwh"]
        .max()
    )
    mids = ((sell_best + buy_best) / 2).to_dict()

    result = []
    for ds in delivery_starts:
        if ds not in mids:
            return None
        result.append(mids[ds])
    return np.array(result, dtype=np.float64)


# ── Day index ─────────────────────────────────────────────────────────────────

def _build_day_index(
    cim: pd.DataFrame,
    auc: pd.DataFrame,
) -> list[tuple]:
    """
    Returns sorted list of (berlin_day, delivery_starts, session_start_utc).

    Only days with exactly 24 delivery hours in both CIM and auction data are
    included.  session_start = 15:00 D-1 in Berlin time, expressed in UTC.
    """
    cim = cim.copy()
    cim["berlin_date"] = (
        cim["delivery_start"]
        .dt.tz_convert("Europe/Berlin")
        .dt.normalize()
    )
    auc_delivery_set = set(auc["delivery_start"].unique())

    days = []
    for day, group in cim.groupby("berlin_date"):
        delivery_starts = sorted(group["delivery_start"].unique().tolist())
        if len(delivery_starts) != 24:
            continue
        if not all(ds in auc_delivery_set for ds in delivery_starts):
            continue
        # 15:00 on D-1 in Berlin time → UTC
        session_berlin = (day - pd.Timedelta(days=1)).replace(
            hour=15, minute=0, second=0
        )
        session_utc = session_berlin.tz_convert("UTC")
        days.append((day, delivery_starts, session_utc))

    return sorted(days, key=lambda x: x[0])


# ── Training loop ─────────────────────────────────────────────────────────────

def _run_phase(
    day_index  : list[tuple],
    cim        : "pd.DataFrame",
    auc        : "pd.DataFrame",
    env        : MultiHourMarketEnv,
    agent      : "REINFORCEAgent",
    n_days     : int,
    reps       : int,
    stride     : int,
    phase_label: str,
    logger     : "TrainingLogger | None" = None,
) -> None:
    """Run `reps` repetitions over the first `n_days` of day_index at `stride`."""
    print(f"\n[{phase_label}]  stride={stride}  reps={reps}")
    print(f"{'ep':>6}  {'rep':>4}  {'reward':>12}  {'loss':>12}  {'steps':>7}")
    print("─" * 56)
    ep_global = 0

    for rep in range(1, reps + 1):
        rep_rewards: list[float] = []

        for ep_idx in range(n_days):
            ep_global += 1

            day, delivery_starts, session_start = day_index[ep_idx]

            day_cim = cim[cim["delivery_start"].isin(delivery_starts)]
            day_auc = auc[auc["delivery_start"].isin(delivery_starts)]

            auction_mids = _day_auction_mids(day_auc, delivery_starts)
            if auction_mids is None:
                continue
            if auction_mids.max() - auction_mids.min() <= 0:
                continue

            state = env.reset(day_cim, delivery_starts, session_start, tick_stride=stride)
            agent.set_episode(
                auction_mids,
                capacity=env.capacity,
                initial_soc=env.initial_soc,
                max_charge=env.max_charge,
                max_dis=env.max_discharge,
            )

            total_reward = 0.0
            n_steps      = 0
            done         = False

            while not done:
                actions, log_prob         = agent.select_actions(state)
                state, reward, done, info = env.step(actions)
                agent.store_step(log_prob, reward, info)
                total_reward += reward
                n_steps      += 1

            loss = agent.update()
            rep_rewards.append(total_reward)

            if logger is not None:
                logger.log_episode(phase_label, total_reward, loss)

            print(f"{ep_global:6d}  {rep:4d}  {total_reward:12.2f}  {loss:12.4f}  {n_steps:7d}")

        # one full pass through training data complete — record rep snapshot
        if logger is not None and rep_rewards:
            logger.log_rep(
                phase         = phase_label,
                local_rep     = rep,
                mean_reward   = float(np.mean(rep_rewards)),
                param_snapshot= agent.param_snapshot(),
            )


def train(args: argparse.Namespace) -> None:
    # ── Hyperparameter initialisation ─────────────────────────────────────────
    if args.seed is not None:
        hp = _sample_hyperparams(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Seed: {args.seed}  — hyperparameters sampled from uniform ranges:")
    else:
        hp = _HP_DEFAULTS.copy()
        print("No seed provided — using fixed default hyperparameters:")
    for k, v in hp.items():
        print(f"  {k:<18} = {v:+.4f}")

    print("\nLoading data …")
    cim, auc = load_all(split="train")

    day_index = _build_day_index(cim, auc)
    n_days    = min(args.days, len(day_index))
    print(f"Days available: {len(day_index)}  |  training on first {n_days}")

    logger = TrainingLogger()

    efficiency = 0.95
    env = MultiHourMarketEnv(
        capacity_mwh             = 200.0,  # paper §VI: 200 MWh storage unit
        max_charge_mw            = 50.0,   # paper §VI: 50 MW
        max_discharge_mw         = 50.0,
        efficiency               = efficiency,
        initial_soc_mwh          = 0.0,
        terminal_penalty_eur_mwh = 0.0,   # paper §V-C: no residual value at end of day
    )
    agent = REINFORCEAgent(lr=args.lr, round_trip_eff=efficiency,
                           n_levels=args.n_levels, **hp)

    # Detect tick spacing to set curriculum strides correctly (§VI-A).
    # minute data (freq="min"): each tick = 60s  → stride 60 ≡ 1-hour step
    # second data (freq="s"):   each tick =  1s  → stride 3600 ≡ 1-hour step
    unique_ts = sorted(cim["timestamp"].unique())
    
    if len(unique_ts) >= 2:
        tick_secs = max(1, round((unique_ts[1] - unique_ts[0]).total_seconds()))
    else:
        tick_secs = 60  # fallback: 1-minute data
    phase_strides = [3600 // tick_secs, 900 // tick_secs, 300 // tick_secs]
    finest_stride = 1
    finest_label = "1-sec" if tick_secs == 1 else "1-min"

    # if we would like to train on multiple different time steps 
    # paper follows hourly, 15mins, 5mins
    if args.curriculum:
        # Three-phase curriculum: hourly → 15-min → 5-min (paper §VI-A)
        for stride, label in zip(phase_strides, ["hourly", "15-min", "5-min"]):
            _run_phase(day_index, cim, auc, env, agent,
                       n_days, args.reps, stride, label, logger=logger)
    else:
        _run_phase(day_index, cim, auc, env, agent,
                   n_days, args.reps, finest_stride, finest_label, logger=logger)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "days_trained"  : n_days,
            "lr"            : args.lr,
            "seed"          : args.seed,
            "round_trip_eff": efficiency,
            "n_levels"      : args.n_levels,
            "hp_init"       : hp,
            "param_snapshot": agent.param_snapshot(),
            "state_dict"    : agent.policy.state_dict(),
        },
        out_path,
    )
    print(f"\nPolicy saved → {out_path}")
    for k, v in agent.param_snapshot().items():
        print(f"  {k:<12} = {v:+.4f}")

    logger.plot(out_dir=args.plot_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train B&P multi-hour REINFORCE policy"
    )
    parser.add_argument(
        "--days", type=int, default=200,
        help="Number of training days (default: 200, matching paper §VI-A)"
    )
    parser.add_argument(
        "--reps", type=int, default=4,
        help="Repetitions over training days per curriculum phase (default: 4, matching paper)"
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Use three-phase frequency curriculum: hourly→15-min→5-min (§VI-A)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam learning rate"
    )
    parser.add_argument(
        "--out", type=str, default="outputs/models/policy_multihour.pt",
        help="Output path for saved policy"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed.  When given, all 11 policy hyperparameters are drawn "
             "uniformly from their natural ranges (see _HP_RANGES).  "
             "Also seeds torch and numpy for reproducibility.  "
             "When omitted, fixed neutral defaults (_HP_DEFAULTS) are used."
    )
    parser.add_argument(
        "--plot-dir", type=str, default="outputs/plots", dest="plot_dir",
        help="Directory for training plots (default: outputs/plots)"
    )
    parser.add_argument(
        "--n-levels", type=int, default=1, dest="n_levels",
        help="Discrete quantity levels per side (paper §III-B). "
             "n=1: binary {0, max}; n=3: {0, max/3, 2*max/3, max}. "
             "Default 1 (effectively binary with our synthetic data depth)."
    )
    train(parser.parse_args())
