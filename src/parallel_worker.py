"""
Fork-based parallel episode workers for training and evaluation.

On Linux (default multiprocessing start_method = 'fork'), child processes
inherit the parent's address space via copy-on-write.  Setting the three
module-level globals in the parent process BEFORE Pool creation means workers
can access the full DataFrames at near-zero cost — no pickling of large data.

Usage pattern in the caller:
    import src.parallel_worker as _pw
    _pw._g_cim       = cim
    _pw._g_auc       = auc
    _pw._g_day_index = day_index
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(_pw.train_episode_worker, tasks)
"""

from __future__ import annotations

import numpy as np
import torch

from src.data_loader import day_auction_mids
from src.environment import MultiHourMarketEnv, N_HOURS
from src.threshold_policy import AlphaPolicy, compute_regimes

# ── Fork-inherited globals (set in parent before Pool creation) ────────────────
_g_cim       = None   # pd.DataFrame — full CIM order book
_g_auc       = None   # pd.DataFrame — intraday auction curves
_g_day_index = None   # list[tuple]  — (day, delivery_starts, session_start)


# ── Training worker ────────────────────────────────────────────────────────────

def train_episode_worker(
    task: dict,
) -> tuple[dict[str, np.ndarray] | None, float, float, int]:
    """
    Run one REINFORCE training episode in a worker process.

    Task keys:
        ep_idx          : int          — index into _g_day_index
        policy_sd       : state_dict   — current policy parameters
        round_trip_eff  : float
        n_levels        : int
        lr              : float
        env_kwargs      : dict         — MultiHourMarketEnv constructor kwargs
        tick_stride     : int
        grad_clip       : float

    Returns:
        (grad_dict, total_reward, loss_val, n_steps)
        grad_dict is None for skipped (invalid) episodes.
    """
    from src.reinforce_trainer import REINFORCEAgent

    ep_idx = task["ep_idx"]
    day, delivery_starts, session_start = _g_day_index[ep_idx]

    day_cim = _g_cim[_g_cim["delivery_start"].isin(set(delivery_starts))]
    day_auc = _g_auc[_g_auc["delivery_start"].isin(set(delivery_starts))]

    auction_mids = day_auction_mids(day_auc, delivery_starts)
    if auction_mids is None or auction_mids.max() - auction_mids.min() <= 0:
        return None, float("nan"), float("nan"), 0

    env   = MultiHourMarketEnv(**task["env_kwargs"])
    agent = REINFORCEAgent(
        lr             = task["lr"],
        round_trip_eff = task["round_trip_eff"],
        n_levels       = task["n_levels"],
    )
    agent.policy.load_state_dict(task["policy_sd"])

    state = env.reset(
        day_cim, delivery_starts, session_start, tick_stride=task["tick_stride"]
    )
    agent.set_episode(
        auction_mids,
        capacity    = env.capacity,
        initial_soc = env.initial_soc,
        max_charge  = env.max_charge,
        max_dis     = env.max_discharge,
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

    loss_val, grad_dict = agent.compute_gradients(grad_clip=task["grad_clip"])
    return grad_dict, total_reward, loss_val, n_steps


# ── Evaluation worker ──────────────────────────────────────────────────────────

def eval_episode_worker(task: dict) -> dict:
    """
    Run one greedy evaluation episode (and optionally the RI-LP benchmark).

    Task keys:
        ep_idx          : int
        policy_sd       : state_dict
        round_trip_eff  : float
        n_levels        : int
        env_kwargs      : dict
        tick_stride     : int
        fast            : bool   — skip per-tick LP in greedy_batch (default True)
        with_ri         : bool   — run RI-LP benchmark (default False)
        record_ticks    : bool   — collect per-tick data for diagnostic plots
        td_sample_every : int    — sample threshold data every N ticks (default 5)

    Returns a dict with keys:
        ep_idx, date, skip, policy_reward, ri_reward,
        final_soc, revenue_per_hour, n_steps,
        tick_records (list[dict] or None),
        td (dict of lists: mu_Y, bid, mu_X, ask, hour).
    """
    ep_idx = task["ep_idx"]
    day, delivery_starts, session_start = _g_day_index[ep_idx]

    day_cim = _g_cim[_g_cim["delivery_start"].isin(set(delivery_starts))]
    day_auc = _g_auc[_g_auc["delivery_start"].isin(set(delivery_starts))]

    auction_mids = day_auction_mids(day_auc, delivery_starts)
    if auction_mids is None or auction_mids.max() - auction_mids.min() <= 0:
        return {"ep_idx": ep_idx, "date": day, "skip": True}

    env = MultiHourMarketEnv(**task["env_kwargs"])

    policy = AlphaPolicy(
        round_trip_eff = task["round_trip_eff"],
        n_levels       = task["n_levels"],
    )
    policy.load_state_dict(task["policy_sd"])
    policy.eval()

    regime = compute_regimes(auction_mids)
    policy.set_regime_info(regime)
    policy.set_env_params(
        env.capacity, env.initial_soc, env.max_charge, env.max_discharge
    )

    # Optional RI-LP benchmark (env.reset is called inside run_ri_benchmark)
    ri_reward = float("nan")
    if task.get("with_ri", False):
        from src.ri_benchmark import run_ri_benchmark
        ri_reward = run_ri_benchmark(
            env, day_cim, delivery_starts, session_start,
            tick_stride=task["tick_stride"],
        )

    # Policy episode
    state           = env.reset(
        day_cim, delivery_starts, session_start, tick_stride=task["tick_stride"]
    )
    revenue_per_hour = np.zeros(N_HOURS)
    td_data          = {"mu_Y": [], "bid": [], "mu_X": [], "ask": [], "hour": []}
    tick_records_raw = [] if task.get("record_ticks", False) else None
    total_reward     = 0.0
    n_steps          = 0
    fast             = task.get("fast", True)
    td_sample_every  = task.get("td_sample_every", 5)
    done             = False

    with torch.no_grad():
        while not done:
            active = state.active_hours

            means: dict = {}
            if active and (
                tick_records_raw is not None or n_steps % td_sample_every == 0
            ):
                means = policy.threshold_means_batch(
                    active,
                    state.best_bid,
                    state.best_ask,
                    state.elapsed_hours,
                    state.v_end_fraction,
                )

            if means and n_steps % td_sample_every == 0:
                for h in active:
                    if h in means and h in state.best_bid and h in state.best_ask:
                        td_data["mu_Y"].append(means[h]["mu_Y"])
                        td_data["bid"].append(state.best_bid[h])
                        td_data["mu_X"].append(means[h]["mu_X"])
                        td_data["ask"].append(state.best_ask[h])
                        td_data["hour"].append(h)

            if tick_records_raw is not None and active and means:
                pos = state.position
                tick_records_raw.append({
                    "elapsed_hours"  : state.elapsed_hours,
                    "n_active"       : len(active),
                    "mean_bid"       : float(np.mean([state.best_bid[h] for h in active])),
                    "mean_ask"       : float(np.mean([state.best_ask[h] for h in active])),
                    "mean_mu_Y"      : float(np.mean([means[h]["mu_Y"] for h in active if h in means])),
                    "mean_mu_X"      : float(np.mean([means[h]["mu_X"] for h in active if h in means])),
                    "total_net_sell" : float(np.sum(np.maximum(0.0, pos))),
                    "total_net_buy"  : float(np.sum(np.maximum(0.0, -pos))),
                })

            actions = policy.greedy_batch(
                state.active_hours,
                state.best_bid,
                state.best_ask,
                state.order_book,
                state.elapsed_hours,
                state.v_end_fraction,
                position   = state.position,
                step_hours = state.step_hours,
                fast       = fast,
            )
            state, reward, done, info = env.step(actions)
            total_reward += reward

            for h, h_info in info.items():
                revenue_per_hour[h] += h_info.get("revenue", 0.0)

            n_steps += 1

    return {
        "ep_idx"          : ep_idx,
        "date"            : day,
        "skip"            : False,
        "policy_reward"   : total_reward,
        "ri_reward"       : ri_reward,
        "final_soc"       : env.implied_soc(),
        "revenue_per_hour": revenue_per_hour,
        "n_steps"         : n_steps,
        "tick_records"    : tick_records_raw,   # list[dict] or None
        "td"              : td_data,
    }
