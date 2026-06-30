"""
Rolling-intrinsic LP benchmark (Bertrand & Papavasiliou 2020, Appendix E).

At each tick t the RI LP solves a linear programme over all remaining active
delivery hours, given the current order book and committed position.

Implements Appendix E faithfully:

1. Full order book depth: one LP variable per bid/ask level per hour.

2. Financial vs physical trade distinction:
   - F_b[h] = prior net buy position for hour h (committed to receive energy)
   - F_s[h] = prior net sell position for hour h (committed to deliver energy)
   Trades that unwind prior commitments are "financial" (no efficiency loss).
   Trades that open new positions are "physical" (efficiency η applies).

LP variables per active hour h:
    s_phy_{h,i} ≥ 0  — physical sell (new discharge) at bid level i
    s_fin_{h,i} ≥ 0  — financial sell (unwind prior buy) at bid level i
    c_phy_{h,j} ≥ 0  — physical buy (new charge) at ask level j
    c_fin_{h,j} ≥ 0  — financial buy (unwind prior sell) at ask level j

Objective (Appendix E, Eq. 25 — maximise revenue):
    Σ_{h,i}  bid_{h,i} × (η × s_phy_{h,i}  +  1 × s_fin_{h,i})
  − Σ_{h,j}  ask_{h,j} × (c_phy_{h,j} / η  +  1 × c_fin_{h,j})

Constraints:
    Σ_i (s_phy_{h,i} + s_fin_{h,i}) ≤ max_dis_step          (power limit)
    Σ_j (c_phy_{h,j} + c_fin_{h,j}) ≤ max_chg_step          (power limit)
    Σ_i  s_fin_{h,i}                ≤ F_b[h]                 (financial sell cap)
    Σ_j  c_fin_{h,j}                ≤ F_s[h]                 (financial buy cap)
    s_phy_{h,i} + s_fin_{h,i}       ≤ Q^b_{h,i}              (level capacity)
    c_phy_{h,j} + c_fin_{h,j}       ≤ Q^s_{h,j}              (level capacity)

    SOC feasibility for each k ∈ {0, …, 23}  (Appendix E, Eq. 33–34):
        Σ_{h≤k} (total_sell_h − total_buy_h) ≤ soc_k
        Σ_{h≤k} (total_buy_h  − total_sell_h) ≤ capacity − soc_k
    where soc_k = initial_soc − cumsum(position)[k].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.environment import MultiHourMarketEnv, N_HOURS

_LP_METHOD = "highs"


def _solve_tick_lp(
    active_hours : list[int],
    best_bids    : dict[int, float],
    best_asks    : dict[int, float],
    implied_soc  : np.ndarray,
    capacity     : float,
    max_dis_step : float,
    max_chg_step : float,
    one_way_eff  : float,
    order_books  : dict[int, dict] | None = None,
    position     : np.ndarray | None = None,
) -> dict[int, float]:
    """
    Solve RI LP for one tick.

    Args:
        order_books : full per-hour order book {h: {"bids": [...], "asks": [...]}}.
                      When None, falls back to best_bids/best_asks as a single level.
        position    : shape-(24,) net committed MWh per hour (+ = net sell).
                      When None, all trades are treated as physical (no financial split).

    Returns:
        {hour: signed_qty}  positive = sell/discharge, negative = buy/charge.
    """
    H = active_hours
    N = len(H)
    if N == 0:
        return {}

    eta = one_way_eff

    # ── Order book levels ─────────────────────────────────────────────────────
    bid_lvls: dict[int, list[tuple[float, float]]] = {}
    ask_lvls: dict[int, list[tuple[float, float]]] = {}
    for h in H:
        if order_books is not None and h in order_books:
            bids = [(float(p), float(q)) for p, q in order_books[h]["bids"] if float(q) > 1e-9]
            asks = [(float(p), float(q)) for p, q in order_books[h]["asks"] if float(q) > 1e-9]
            bid_lvls[h] = bids if bids else [(best_bids[h], max_dis_step)]
            ask_lvls[h] = asks if asks else [(best_asks[h], max_chg_step)]
        else:
            bid_lvls[h] = [(best_bids[h], max_dis_step)]
            ask_lvls[h] = [(best_asks[h], max_chg_step)]

    # ── Financial position caps ───────────────────────────────────────────────
    # F_b[h]: prior net buy → can sell financially up to F_b[h] MWh
    # F_s[h]: prior net sell → can buy financially up to F_s[h] MWh
    if position is not None:
        F_b = {h: float(max(0.0, -position[h])) for h in H}
        F_s = {h: float(max(0.0,  position[h])) for h in H}
    else:
        F_b = {h: 0.0 for h in H}
        F_s = {h: 0.0 for h in H}

    # ── Variable layout ───────────────────────────────────────────────────────
    # x = [s_phy_{h,i}] ++ [s_fin_{h,i}] ++ [c_phy_{h,j}] ++ [c_fin_{h,j}]
    s_phy_off: dict[int, int] = {}
    s_fin_off: dict[int, int] = {}
    c_phy_off: dict[int, int] = {}
    c_fin_off: dict[int, int] = {}

    idx = 0
    for h in H:
        s_phy_off[h] = idx;  idx += len(bid_lvls[h])
    for h in H:
        s_fin_off[h] = idx;  idx += len(bid_lvls[h])
    for h in H:
        c_phy_off[h] = idx;  idx += len(ask_lvls[h])
    for h in H:
        c_fin_off[h] = idx;  idx += len(ask_lvls[h])
    n_vars = idx

    # ── Objective (minimise negative revenue) ────────────────────────────────
    c_obj = np.zeros(n_vars)
    for h in H:
        for i, (p, _) in enumerate(bid_lvls[h]):
            c_obj[s_phy_off[h] + i] = -p * eta   # physical sell: revenue × η
            c_obj[s_fin_off[h] + i] = -p          # financial sell: full revenue
        for j, (p, _) in enumerate(ask_lvls[h]):
            c_obj[c_phy_off[h] + j] =  p / eta   # physical buy: cost / η
            c_obj[c_fin_off[h] + j] =  p          # financial buy: full cost

    # ── Variable bounds ───────────────────────────────────────────────────────
    # Level capacity enforced jointly via constraints below; individual bounds
    # keep the LP well-posed but the joint constraint is the binding one.
    bounds = []
    for h in H:
        for _, q in bid_lvls[h]:
            bounds.append((0.0, q))   # s_phy
    for h in H:
        for _, q in bid_lvls[h]:
            bounds.append((0.0, q))   # s_fin
    for h in H:
        for _, q in ask_lvls[h]:
            bounds.append((0.0, q))   # c_phy
    for h in H:
        for _, q in ask_lvls[h]:
            bounds.append((0.0, q))   # c_fin

    # ── Constraints ──────────────────────────────────────────────────────────
    total_bid = sum(len(bid_lvls[h]) for h in H)
    total_ask = sum(len(ask_lvls[h]) for h in H)

    # Row blocks:
    #  0      .. N-1                : sell power limits
    #  N      .. 2N-1               : buy power limits
    #  2N     .. 3N-1               : financial sell caps
    #  3N     .. 4N-1               : financial buy caps
    #  4N     .. 4N+total_bid-1     : level capacity (sell)
    #  4N+B   .. 4N+B+total_ask-1   : level capacity (buy)   B=total_bid
    #  4N+B+A .. 4N+B+A+24-1        : SOC >= 0               A=total_ask
    #  4N+B+A+24 .. 4N+B+A+48-1    : SOC <= cap

    B, A = total_bid, total_ask
    soc0_start = 4 * N + B + A
    n_con = soc0_start + 2 * N_HOURS

    A_ub = np.zeros((n_con, n_vars))
    b_ub = np.zeros(n_con)

    row = 0

    # Sell power limits
    for h in H:
        for i in range(len(bid_lvls[h])):
            A_ub[row, s_phy_off[h] + i] = 1.0
            A_ub[row, s_fin_off[h] + i] = 1.0
        b_ub[row] = max_dis_step
        row += 1

    # Buy power limits
    for h in H:
        for j in range(len(ask_lvls[h])):
            A_ub[row, c_phy_off[h] + j] = 1.0
            A_ub[row, c_fin_off[h] + j] = 1.0
        b_ub[row] = max_chg_step
        row += 1

    # Financial sell caps
    for h in H:
        for i in range(len(bid_lvls[h])):
            A_ub[row, s_fin_off[h] + i] = 1.0
        b_ub[row] = F_b[h]
        row += 1

    # Financial buy caps
    for h in H:
        for j in range(len(ask_lvls[h])):
            A_ub[row, c_fin_off[h] + j] = 1.0
        b_ub[row] = F_s[h]
        row += 1

    # Level capacity (sell): s_phy_{h,i} + s_fin_{h,i} <= Q^b_{h,i}
    for h in H:
        for i, (_, q) in enumerate(bid_lvls[h]):
            A_ub[row, s_phy_off[h] + i] = 1.0
            A_ub[row, s_fin_off[h] + i] = 1.0
            b_ub[row] = q
            row += 1

    # Level capacity (buy): c_phy_{h,j} + c_fin_{h,j} <= Q^s_{h,j}
    for h in H:
        for j, (_, q) in enumerate(ask_lvls[h]):
            A_ub[row, c_phy_off[h] + j] = 1.0
            A_ub[row, c_fin_off[h] + j] = 1.0
            b_ub[row] = q
            row += 1

    # SOC constraints (Eq. 33–34)
    for k in range(N_HOURS):
        r_lo = soc0_start + k
        r_hi = soc0_start + N_HOURS + k
        for h in H:
            if h <= k:
                for i in range(len(bid_lvls[h])):
                    A_ub[r_lo, s_phy_off[h] + i] =  1.0   # sell reduces SOC
                    A_ub[r_lo, s_fin_off[h] + i] =  1.0
                    A_ub[r_hi, s_phy_off[h] + i] = -1.0
                    A_ub[r_hi, s_fin_off[h] + i] = -1.0
                for j in range(len(ask_lvls[h])):
                    A_ub[r_lo, c_phy_off[h] + j] = -1.0   # buy increases SOC
                    A_ub[r_lo, c_fin_off[h] + j] = -1.0
                    A_ub[r_hi, c_phy_off[h] + j] =  1.0
                    A_ub[r_hi, c_fin_off[h] + j] =  1.0
        b_ub[r_lo] = float(implied_soc[k])
        b_ub[r_hi] = float(capacity - implied_soc[k])

    # ── Solve ─────────────────────────────────────────────────────────────────
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=_LP_METHOD)

    if result.status != 0:
        return {h: 0.0 for h in H}

    x = result.x
    actions: dict[int, float] = {}
    for h in H:
        total_sell = sum(
            float(x[s_phy_off[h] + i]) + float(x[s_fin_off[h] + i])
            for i in range(len(bid_lvls[h]))
        )
        total_buy = sum(
            float(x[c_phy_off[h] + j]) + float(x[c_fin_off[h] + j])
            for j in range(len(ask_lvls[h]))
        )
        if total_sell > total_buy + 1e-9:
            actions[h] = total_sell
        elif total_buy > total_sell + 1e-9:
            actions[h] = -total_buy
        else:
            actions[h] = 0.0

    return actions


def run_ri_benchmark(
    env             : MultiHourMarketEnv,
    day_cim         : pd.DataFrame,
    delivery_starts : list[pd.Timestamp],
    session_start   : pd.Timestamp,
    tick_stride     : int = 1,
    verbose         : bool = False,
) -> float:
    """
    Run one episode using the rolling-intrinsic LP strategy (Appendix E).

    Re-solves the LP at every tick using full order book depth, current best
    bid/ask prices, and the updated cumulative position (for financial/physical
    trade distinction).  Executes LP-optimal quantities through the environment.

    Returns:
        total_reward (EUR) for the episode.
    """
    state = env.reset(day_cim, delivery_starts, session_start, tick_stride=tick_stride)
    total_reward = 0.0
    n_steps = 0

    while state.active_hours:
        implied_soc = env.initial_soc - np.cumsum(state.position)
        step_hours  = state.step_hours

        actions = _solve_tick_lp(
            active_hours = state.active_hours,
            best_bids    = state.best_bid,
            best_asks    = state.best_ask,
            implied_soc  = implied_soc,
            capacity     = env.capacity,
            max_dis_step = env.max_discharge * step_hours,
            max_chg_step = env.max_charge    * step_hours,
            one_way_eff  = env.one_way_eff,
            order_books  = state.order_book,
            position     = state.position,
        )

        state, reward, done, _ = env.step(actions)
        total_reward += reward
        n_steps      += 1

        if done:
            break

    if verbose:
        print(f"  RI-LP steps={n_steps}  total_reward={total_reward:.2f} EUR")

    return total_reward
