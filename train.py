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
from datetime import datetime
from pathlib import Path

import multiprocessing

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import build_day_index, day_auction_mids, load_all
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


# ── Training loop ─────────────────────────────────────────────────────────────

def _apply_averaged_gradients(
    agent     : "REINFORCEAgent",
    grad_dicts: list[dict[str, np.ndarray]],
) -> None:
    """Average REINFORCE gradients from N workers and apply one optimizer step."""
    agent.optimizer.zero_grad()
    for name, param in agent.policy.named_parameters():
        grads = [gd[name] for gd in grad_dicts if gd is not None]
        if grads:
            param.grad = torch.as_tensor(np.mean(grads, axis=0), dtype=param.dtype)
    agent.optimizer.step()


def _is_plateaued(rep_means: list[float], window: int, tol: float) -> bool:
    """
    True once the rolling per-rep mean reward has stopped improving by more
    than `tol` (relative) between the previous and current `window`-rep
    blocks (paper §VI-A, footnote 7: switch frequency once profit stabilises).
    """
    if len(rep_means) < 2 * window:
        return False
    prev = float(np.mean(rep_means[-2 * window : -window]))
    curr = float(np.mean(rep_means[-window:]))
    denom = max(abs(prev), 1.0)
    return (curr - prev) / denom < tol


def _run_phase_sequential(
    day_index  : list[tuple],
    cim        : "pd.DataFrame",
    auc        : "pd.DataFrame",
    env        : MultiHourMarketEnv,
    agent      : "REINFORCEAgent",
    n_days     : int,
    reps       : int | float,
    stride     : int,
    phase_label: str,
    logger     : "TrainingLogger | None" = None,
    plateau_window    : int   = 10,
    plateau_tol       : float = 0.02,
    max_reps_per_phase: int   = 500,
) -> None:
    """
    Original sequential training: one episode → one gradient update.

    If `reps` is finite, runs exactly that many passes (original behaviour).
    If `reps` is inf, runs until the rolling per-rep mean reward stabilises
    (§VI-A footnote 7), capped at `max_reps_per_phase` as a safety net.
    """
    ep_global = 0
    rep_means: list[float] = []
    rep = 0

    while True:
        rep += 1
        if reps != float("inf") and rep > reps:
            break
        if reps == float("inf") and rep > max_reps_per_phase:
            print(f"[{phase_label}] reached max_reps_per_phase={max_reps_per_phase} "
                  f"without detecting a plateau; stopping phase.")
            break

        rep_rewards: list[float] = []

        for ep_idx in range(n_days):
            ep_global += 1

            day, delivery_starts, session_start = day_index[ep_idx]

            day_cim = cim[cim["delivery_start"].isin(delivery_starts)]
            day_auc = auc[auc["delivery_start"].isin(delivery_starts)]

            auction_mids = day_auction_mids(day_auc, delivery_starts)
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

        if rep_rewards:
            mean_reward = float(np.mean(rep_rewards))
            rep_means.append(mean_reward)
            if logger is not None:
                logger.log_rep(
                    phase         = phase_label,
                    local_rep     = rep,
                    mean_reward   = mean_reward,
                    param_snapshot= agent.param_snapshot(),
                )

        if reps == float("inf") and _is_plateaued(rep_means, plateau_window, plateau_tol):
            print(f"[{phase_label}] profit stabilised after {rep} reps "
                  f"(window={plateau_window}, tol={plateau_tol:.1%}) — advancing phase.")
            break


def _run_phase_parallel(
    day_index      : list[tuple],
    cim            : "pd.DataFrame",
    auc            : "pd.DataFrame",
    env            : MultiHourMarketEnv,
    agent          : "REINFORCEAgent",
    n_days         : int,
    reps           : int | float,
    stride         : int,
    phase_label    : str,
    logger         : "TrainingLogger | None",
    workers        : int,
    round_trip_eff : float,
    lr             : float,
    plateau_window    : int   = 10,
    plateau_tol       : float = 0.02,
    max_reps_per_phase: int   = 500,
) -> None:
    """
    Parallel training: N episodes run simultaneously → gradients averaged →
    one optimizer step per batch.  Equivalent to mini-batch REINFORCE with
    batch size = workers (lower variance than single-episode updates).
    """
    import src.parallel_worker as _pw

    # Expose DataFrames to workers via fork-inherited module globals.
    # On Linux (fork), children read these at zero copy cost.
    _pw._g_cim       = cim
    _pw._g_auc       = auc
    _pw._g_day_index = day_index

    env_kwargs = dict(
        capacity_mwh             = env.capacity,
        max_charge_mw            = env.max_charge,
        max_discharge_mw         = env.max_discharge,
        efficiency               = env.one_way_eff ** 2,
        initial_soc_mwh          = env.initial_soc,
        terminal_penalty_eur_mwh = env.terminal_penalty,
    )

    ep_global = 0
    rep_means: list[float] = []
    rep = 0

    with multiprocessing.get_context("fork").Pool(workers) as pool:
        while True:
            rep += 1
            if reps != float("inf") and rep > reps:
                break
            if reps == float("inf") and rep > max_reps_per_phase:
                print(f"[{phase_label}] reached max_reps_per_phase={max_reps_per_phase} "
                      f"without detecting a plateau; stopping phase.")
                break

            rep_rewards: list[float] = []

            for batch_start in range(0, n_days, workers):
                batch_idx = list(range(batch_start, min(batch_start + workers, n_days)))

                # Broadcast current parameters to workers (12 scalar tensors — tiny)
                policy_sd = {
                    k: v.detach().clone()
                    for k, v in agent.policy.state_dict().items()
                }

                tasks = [
                    {
                        "ep_idx"        : ep_idx,
                        "policy_sd"     : policy_sd,
                        "round_trip_eff": round_trip_eff,
                        "n_levels"      : agent.policy.n_levels,
                        "lr"            : lr,
                        "env_kwargs"    : env_kwargs,
                        "tick_stride"   : stride,
                        "grad_clip"     : 1.0,
                    }
                    for ep_idx in batch_idx
                ]

                results = pool.map(_pw.train_episode_worker, tasks)

                valid = [
                    (g, r, l, s) for g, r, l, s in results if g is not None
                ]
                ep_global += len(batch_idx)

                if not valid:
                    continue

                grad_dicts, rewards, losses, steps = zip(*valid)
                _apply_averaged_gradients(agent, list(grad_dicts))

                for r, l, s in zip(rewards, losses, steps):
                    if logger is not None:
                        logger.log_episode(phase_label, r, l)
                    rep_rewards.append(r)
                    print(
                        f"{ep_global:6d}  {rep:4d}  {r:12.2f}  {l:12.4f}  {s:7d}"
                    )

            if rep_rewards:
                mean_reward = float(np.mean(rep_rewards))
                rep_means.append(mean_reward)
                if logger is not None:
                    logger.log_rep(
                        phase         = phase_label,
                        local_rep     = rep,
                        mean_reward   = mean_reward,
                        param_snapshot= agent.param_snapshot(),
                    )

            if reps == float("inf") and _is_plateaued(rep_means, plateau_window, plateau_tol):
                print(f"[{phase_label}] profit stabilised after {rep} reps "
                      f"(window={plateau_window}, tol={plateau_tol:.1%}) — advancing phase.")
                break


def _run_phase(
    day_index  : list[tuple],
    cim        : "pd.DataFrame",
    auc        : "pd.DataFrame",
    env        : MultiHourMarketEnv,
    agent      : "REINFORCEAgent",
    n_days     : int,
    reps       : int | float,
    stride     : int,
    phase_label: str,
    logger     : "TrainingLogger | None" = None,
    workers    : int = 1,
    round_trip_eff: float = 1.0,
    lr         : float = 1e-3,
    plateau_window    : int   = 10,
    plateau_tol       : float = 0.02,
    max_reps_per_phase: int   = 500,
) -> None:
    """
    Run `reps` repetitions over the first `n_days` of day_index at `stride`.

    reps=inf trains until the rolling per-rep mean reward stabilises instead
    of a fixed count (§VI-A footnote 7), capped at `max_reps_per_phase`.
    """
    suffix = f"  workers={workers}" if workers > 1 else ""
    if reps == float("inf"):
        reps_desc = (f"reps=inf (until plateau: window={plateau_window}, "
                     f"tol={plateau_tol:.1%}, max={max_reps_per_phase})")
    else:
        reps_desc = f"reps={reps}"
    print(f"\n[{phase_label}]  stride={stride}  {reps_desc}{suffix}")
    print(f"{'ep':>6}  {'rep':>4}  {'reward':>12}  {'loss':>12}  {'steps':>7}")
    print("─" * 56)

    if workers > 1:
        _run_phase_parallel(
            day_index, cim, auc, env, agent, n_days, reps, stride,
            phase_label, logger, workers, round_trip_eff, lr,
            plateau_window, plateau_tol, max_reps_per_phase,
        )
    else:
        _run_phase_sequential(
            day_index, cim, auc, env, agent, n_days, reps, stride,
            phase_label, logger,
            plateau_window=plateau_window, plateau_tol=plateau_tol,
            max_reps_per_phase=max_reps_per_phase,
        )


def _write_hparams(path: Path, args: argparse.Namespace, hp: dict, run_dir: Path) -> None:
    """Write a human-readable record of all training config to hparams.txt."""
    lines = [
        f"Run directory : {run_dir}",
        f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== Training Configuration ===",
        f"  days           : {args.days}",
        f"  reps           : {args.reps}",
        f"  lr             : {args.lr}",
        f"  seed           : {args.seed if args.seed is not None else 'None (fixed defaults)'}",
        f"  n_levels       : {args.n_levels}",
        f"  curriculum     : {args.curriculum}",
        f"  workers        : {args.workers}",
    ]
    if args.reps == float("inf"):
        lines += [
            f"  plateau_window : {args.plateau_window}",
            f"  plateau_tol    : {args.plateau_tol}",
            f"  max_reps/phase : {args.max_reps_per_phase}",
        ]
    lines += [
        "",
        "=== Policy Hyperparameters (initial) ===",
    ]
    for k, v in hp.items():
        lines.append(f"  {k:<18} : {v:+.4f}")
    lines += [
        "",
        "=== Environment ===",
        "  capacity_mwh        : 200.0",
        "  max_charge_mw       : 50.0",
        "  max_discharge_mw    : 50.0",
        "  one_way_efficiency  : 0.95  (round-trip = 0.9025)",
        "  initial_soc_mwh     : 0.0",
        "  terminal_penalty    : 0.0",
    ]
    path.write_text("\n".join(lines) + "\n")


def train(args: argparse.Namespace) -> None:
    # ── Run directory ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        run_dir  = Path("outputs/runs") / timestamp
        out_path = run_dir / "model.pt"
        plot_dir = run_dir / "training"
    else:
        out_path = Path(args.out)
        run_dir  = out_path.parent
        plot_dir = Path(args.plot_dir) if args.plot_dir else run_dir / "training"

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory : {run_dir}")

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

    _write_hparams(run_dir / "hparams.txt", args, hp, run_dir)
    print(f"Hyperparameters saved → {run_dir / 'hparams.txt'}")

    print("\nLoading data …")
    cim, auc = load_all(split="train")

    day_index = build_day_index(cim, auc)
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

    common = dict(
        workers=args.workers, round_trip_eff=efficiency, lr=args.lr,
        plateau_window=args.plateau_window, plateau_tol=args.plateau_tol,
        max_reps_per_phase=args.max_reps_per_phase,
    )

    if args.curriculum:
        # Three-phase curriculum: hourly → 15-min → 5-min (paper §VI-A)
        for stride, label in zip(phase_strides, ["hourly", "15-min", "5-min"]):
            _run_phase(day_index, cim, auc, env, agent,
                       n_days, args.reps, stride, label, logger=logger, **common)
    else:
        _run_phase(day_index, cim, auc, env, agent,
                   n_days, args.reps, finest_stride, finest_label,
                   logger=logger, **common)

    # ── Save ──────────────────────────────────────────────────────────────────
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

    logger.plot(out_dir=plot_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _reps_type(value: str) -> int | float:
    """Accept a positive int, or 'inf' to train each phase until profit plateaus."""
    if value.strip().lower() in ("inf", "infinity"):
        return float("inf")
    return int(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train B&P multi-hour REINFORCE policy"
    )
    parser.add_argument(
        "--days", type=int, default=200,
        help="Number of training days (default: 200, matching paper §VI-A)"
    )
    parser.add_argument(
        "--reps", type=_reps_type, default=4,
        help="Repetitions over training days per curriculum phase (default: 4). "
             "Pass 'inf' to instead train each phase until profit stabilises "
             "(paper §VI-A footnote 7: switch frequency once profit plateaus), "
             "bounded by --max-reps-per-phase as a safety cap."
    )
    parser.add_argument(
        "--plateau-window", type=int, default=10, dest="plateau_window",
        help="Number of reps averaged together when checking for profit "
             "stabilisation. Only used with --reps inf. Default: 10."
    )
    parser.add_argument(
        "--plateau-tol", type=float, default=0.02, dest="plateau_tol",
        help="Relative improvement threshold (of the previous window's mean "
             "reward) below which profit is considered stabilised. Only used "
             "with --reps inf. Default: 0.02 (2%%)."
    )
    parser.add_argument(
        "--max-reps-per-phase", type=int, default=500, dest="max_reps_per_phase",
        help="Safety cap on reps per phase when --reps inf never detects a "
             "plateau. Default: 500."
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
        "--out", type=str, default=None,
        help="Output path for saved model (default: outputs/runs/<timestamp>/model.pt)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed.  When given, all 11 policy hyperparameters are drawn "
             "uniformly from their natural ranges (see _HP_RANGES).  "
             "Also seeds torch and numpy for reproducibility.  "
             "When omitted, fixed neutral defaults (_HP_DEFAULTS) are used."
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None, dest="plot_dir",
        help="Directory for training plots (default: <run_dir>/training/)"
    )
    parser.add_argument(
        "--n-levels", type=int, default=1, dest="n_levels",
        help="Discrete quantity levels per side (paper §III-B). "
             "n=1: binary {0, max}; n=3: {0, max/3, 2*max/3, max}. "
             "Default 1 (effectively binary with our synthetic data depth)."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes for episode collection.  "
             "N>1 runs N episodes concurrently and averages their REINFORCE "
             "gradients before each optimizer step (mini-batch REINFORCE). "
             "Uses fork-based multiprocessing — Linux/WSL2 only.  "
             "Recommended: set to the number of physical CPU cores (e.g. 8 or 16). "
             "Default: 1 (sequential, original behaviour)."
    )
    train(parser.parse_args())
