"""
B&P alpha-parameterised threshold policy (Bertrand & Papavasiliou 2020, §V).

Multi-hour episode version: episode = one full trading day (24 delivery hours).

Action space (§III-B):
    At each tick, for each delivery hour d, the agent selects a signed quantity
    from n_levels+1 sell levels {0, q/n, …, q} and n_levels+1 buy levels.
    Positive quantities = discharge/sell, negative = charge/buy, 0 = idle.

Regime structure (§III-B, §V-B through §V-E):
    The 24-hour D-1 auction MID curve is split into:
      Buy regimes  — segments between consecutive local maxima.
                     Each buy regime k has p_min,k = min price in segment;
                     T_s,k = delivery time (hours from 15:00 D-1) of p_min,k.
      Sell regimes — segments between consecutive local minima.
                     Each sell regime k has p_max,k = max price in segment;
                     T_b,k = delivery time of p_max,k.

Threshold means for delivery hour h at trading time t_hours:

    Step 1 — auction anchor (§V-B):
        μ_X = p_min[h] + α₁^s · (p_max[h] − p_min[h])
        μ_Y = p_max[h] − α₁^b · (p_max[h] − p_min[h])

    Step 2 — position adjustment (§V-C):
        μ_X ← μ_X − α₂^s · v_end_fraction
        μ_Y ← μ_Y − α₂^b · v_end_fraction

    Step 3 — urgency (§V-D, Eq. 4–5):
        μ_X ← μ_X + α₃^s · (range/2) · exp(α₄^s · (t − T_s[h]))
        μ_Y ← μ_Y − α₃^b · (range/2) · exp(α₄^b · (t − T_b[h]))
        Full exponent α₄ · Δt clamped to ≤ 5 to prevent overflow.

    Step 4 — rolling-intrinsic reference (§V-E, Appendix D):
        Auxiliary Gaussian probability reallocation.
        Probability mass is transferred from levels > RI boundary to the boundary.
        NOT a mean shift — probabilities are computed per-level, not per-µ.

    Step 5 — efficiency adjustment (§V-G, no new parameters):
        Main (financial) distribution:  N(η·μ_Y, σ_Y) sell /  N(μ_X/η, σ_X) buy.
        Auxiliary (physical) distribution: N(μ_Y/η², σ_Y) sell / N(η²·μ_X, σ_X) buy.
        The RI boundary from Step 4 selects which distribution applies per level.
        α₅^b / α₅^s (§V-E) provide an additional learned shift on the physical auxiliary.

Stochastic threshold policy (§IV-B, Appendix C):
    X ~ N(μ_X, exp(σ_X)²)   [buy threshold]
    Y ~ N(μ_Y, exp(σ_Y)²)   [sell threshold]

    Sell level determined by comparing Y against bid-curve midpoint prices
    (from top of book downward); buy level by comparing X against ask-curve
    midpoint prices.  Sell takes priority over buy.

Log-probability for REINFORCE (discrete action, §IV-A, Appendix C):
    Uses CDF differences rather than continuous density at sampled threshold.

    P(sell level k)  = Φ(p_sell_mid[k−1]; μ_Y, σ_Y) − Φ(p_sell_mid[k]; μ_Y, σ_Y)
    P(sell 0)        = 1 − Φ(p_sell_mid[0]; μ_Y, σ_Y)

    P(buy level k)   = Φ(p_buy_mid[k]; μ_X, σ_X) − Φ(p_buy_mid[k−1]; μ_X, σ_X)
    P(buy 0)         = Φ(p_buy_mid[0]; μ_X, σ_X)

    For a combined sell-priority action:
      P(action = sell q_k) = P(sell level = k)
      P(action = buy  q_k) = P(sell level = 0) × P(buy level = k)
      P(action = idle)     = P(sell level = 0) × P(buy level = 0)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from src.environment import BookLevel
from src.ri_benchmark import _solve_tick_lp

_STEP_HOURS = 1 / 60


# ── Order book helpers ────────────────────────────────────────────────────────

def book_price_at_qty(book: list[BookLevel], qty: float) -> float:
    """
    Return the marginal order book price at cumulative quantity qty.

    For a bids book (price DESC) this gives the bid price at depth qty.
    For an asks book (price ASC)  this gives the ask price at depth qty.
    Falls back to the last level's price when the book is shallower than qty.
    """
    cum = 0.0
    for price, level_qty in book:
        cum += level_qty
        if cum >= qty:
            return float(price)
    return float(book[-1][0]) if book else 0.0


def sell_midpoint_prices(
    bids     : list[BookLevel],
    n_levels : int,
    max_qty  : float,
) -> np.ndarray:
    """
    Midpoint bid prices for n_levels non-zero sell levels.

    Sell levels: {q_k = k*max_qty/n_levels  for k = 0..n_levels}.
    Midpoints:   q_mid_k = (q_{k−1} + q_k) / 2  for k = 1..n_levels.
    Returns:     shape (n_levels,), prices DESCENDING (best bid first).
    """
    q_levels = np.linspace(0.0, max_qty, n_levels + 1)
    q_mids   = (q_levels[:-1] + q_levels[1:]) / 2.0
    return np.array([book_price_at_qty(bids, qm) for qm in q_mids], dtype=np.float64)


def buy_midpoint_prices(
    asks     : list[BookLevel],
    n_levels : int,
    max_qty  : float,
) -> np.ndarray:
    """
    Midpoint ask prices for n_levels non-zero buy levels.
    Returns: shape (n_levels,), prices ASCENDING (cheapest ask first).
    """
    q_levels = np.linspace(0.0, max_qty, n_levels + 1)
    q_mids   = (q_levels[:-1] + q_levels[1:]) / 2.0
    return np.array([book_price_at_qty(asks, qm) for qm in q_mids], dtype=np.float64)



def _sell_level_probs_ri(
    p_mids_desc  : torch.Tensor,   # shape (n_levels,), DESCENDING bid mid prices
    dist_Y       : Normal,          # main sell distribution: N(η·µ_Y, σ_Y) [financial, §V-G]
    dist_Y_phys  : Normal,          # physical sell distribution: N(µ_Y/η², σ_Y) [§V-G]
    alpha_b5     : torch.Tensor,    # RI auxiliary additive shift (§V-E)
    ri_level     : int,             # sell level RI would accept (0 = RI sells nothing)
    n_levels     : int,
) -> torch.Tensor:
    """
    Full sell probability vector using auxiliary Gaussian reallocation (§V-E + §V-G).

    Returns shape (n_levels + 1,): P(sell 0), P(sell q_1), …, P(sell q_n).

    Main distribution  dist_Y      = N(η·µ_Y, σ_Y)       — financial (§V-G baseline).
    Auxiliary          dist_Y_phys = N(µ_Y/η², σ_Y)      — physical efficiency (§V-G).
    The auxiliary mean is further shifted by α₅^b for RI correction (§V-E).

    Levels ≤ ri_level: use dist_Y (financial/main).
    Levels > ri_level: use N(µ_Y_phys + α₅^b, σ_Y) (physical + RI correction).
    Level = ri_level absorbs probability mass from above-RI levels.

    When efficiency = 1: dist_Y = dist_Y_phys → reduces to original §V-E behaviour.
    """
    dist_aux = Normal(dist_Y_phys.loc + alpha_b5, dist_Y.scale)
    n        = n_levels

    probs: list[torch.Tensor] = []

    # Decide which distribution governs the level-1 upper boundary
    # (= governs level 0's complement)
    dist_lvl0_complement = dist_Y if ri_level >= 1 else dist_aux
    probs.append((1.0 - dist_lvl0_complement.cdf(p_mids_desc[0])).clamp(min=0.0))

    for k in range(1, n + 1):
        # Distribution for the upper boundary price of level k (p_mids[k-1])
        dist_upper = dist_Y   if k <= ri_level else dist_aux
        # Distribution for the lower boundary price of level k (p_mids[k])
        dist_lower = dist_Y   if k <  ri_level else dist_aux

        if k == n:
            # Last level: no lower boundary
            probs.append(dist_upper.cdf(p_mids_desc[k - 1]).clamp(min=0.0))
        else:
            p_k = dist_upper.cdf(p_mids_desc[k - 1]) - dist_lower.cdf(p_mids_desc[k])
            probs.append(p_k.clamp(min=0.0))

    raw = torch.stack(probs)
    return (raw / raw.sum().clamp(min=1e-8)).clamp(min=1e-8)   # renormalise for numerical safety


def _buy_level_probs_ri(
    p_mids_asc  : torch.Tensor,   # shape (n_levels,), ASCENDING ask mid prices
    dist_X      : Normal,          # main buy distribution: N(µ_X/η, σ_X) [financial, §V-G]
    dist_X_phys : Normal,          # physical buy distribution: N(η²·µ_X, σ_X) [§V-G]
    alpha_s5    : torch.Tensor,    # RI auxiliary shift parameter (§V-E)
    ri_level    : int,             # buy level RI would accept (0 = RI buys nothing)
    n_levels    : int,
) -> torch.Tensor:
    """
    Full buy probability vector using auxiliary Gaussian reallocation (§V-E + §V-G).

    Returns shape (n_levels + 1,): P(buy 0), P(buy q_1), …, P(buy q_n).

    Main distribution  dist_X      = N(µ_X/η, σ_X)       — financial (§V-G baseline).
    Auxiliary          dist_X_phys = N(η²·µ_X, σ_X)      — physical efficiency (§V-G).
    The auxiliary mean is further shifted by −α₅^s for RI correction (§V-E).

    When efficiency = 1: dist_X = dist_X_phys → reduces to original §V-E behaviour.
    """
    dist_aux = Normal(dist_X_phys.loc - alpha_s5, dist_X.scale)
    n        = n_levels

    probs: list[torch.Tensor] = []

    dist_lvl0_complement = dist_X if ri_level >= 1 else dist_aux
    probs.append(dist_lvl0_complement.cdf(p_mids_asc[0]).clamp(min=0.0))

    for k in range(1, n + 1):
        dist_upper = dist_X   if k <= ri_level else dist_aux
        dist_lower = dist_X   if k <  ri_level else dist_aux

        if k == n:
            probs.append((1.0 - dist_upper.cdf(p_mids_asc[k - 1])).clamp(min=0.0))
        else:
            p_k = dist_upper.cdf(p_mids_asc[k]) - dist_lower.cdf(p_mids_asc[k - 1])
            probs.append(p_k.clamp(min=0.0))

    raw = torch.stack(probs)
    return (raw / raw.sum().clamp(min=1e-8)).clamp(min=1e-8)


# ── Feasibility reweighting (§V-F imbalance prevention) ──────────────────────

def _reweight_feasibility(
    probs           : torch.Tensor,   # shape (n_levels + 1,)
    max_feasible_lvl: int,             # highest level with feasible quantity
) -> torch.Tensor:
    """
    Redirect probability mass from infeasible levels (> max_feasible_lvl) to the
    highest feasible level, implementing the "large constant M" reallocation of
    §V-F.  All infeasible levels are zeroed; mass is conserved.

    Differentiable w.r.t. probs: the infeasible mass is summed and added to
    probs[max_feasible_lvl], so gradients flow through both.
    """
    n = probs.shape[0] - 1
    if max_feasible_lvl >= n:
        return probs  # all levels feasible

    if max_feasible_lvl < 0:
        # Completely infeasible: collapse everything to level 0 (idle)
        return torch.cat([probs.sum().unsqueeze(0),
                          torch.zeros(n, dtype=probs.dtype)])

    feasible  = probs[:max_feasible_lvl + 1]
    infeasible_mass = probs[max_feasible_lvl + 1:].sum()

    boundary = feasible[max_feasible_lvl] + infeasible_mass
    feasible_adjusted = torch.cat([
        feasible[:max_feasible_lvl].clamp(min=1e-8),
        boundary.clamp(min=1e-8).unsqueeze(0),
    ])
    return torch.cat([
        feasible_adjusted,
        torch.zeros(n - max_feasible_lvl, dtype=probs.dtype),
    ])


# ── Rolling-intrinsic reference — LP version (§V-E, Appendix E) ─────────────

def _rolling_intrinsic_lp_levels(
    active_hours : list[int],
    best_bids    : dict[int, float],
    best_asks    : dict[int, float],
    position     : np.ndarray,
    capacity     : float,
    initial_soc  : float,
    max_charge_mw: float,
    max_dis_mw   : float,
    one_way_eff  : float,
    step_hours   : float,
    n_levels     : int,
    order_books  : dict[int, dict] | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    LP-based rolling-intrinsic reference (paper §V-E, Appendix E).

    Solves the tick LP from Appendix E with full order book depth and
    financial/physical trade distinction; maps optimal quantities to integer
    levels {0, …, n_levels} via ceiling scaling.

    Returns (ri_sell_levels, ri_buy_levels): per-hour integer level dicts.
    """
    implied_soc  = initial_soc - np.cumsum(position)
    max_dis_step = max_dis_mw    * step_hours
    max_chg_step = max_charge_mw * step_hours

    lp_qty = _solve_tick_lp(
        active_hours = active_hours,
        best_bids    = best_bids,
        best_asks    = best_asks,
        implied_soc  = implied_soc,
        capacity     = capacity,
        max_dis_step = max_dis_step,
        max_chg_step = max_chg_step,
        one_way_eff  = one_way_eff,
        order_books  = order_books,
        position     = position,
    )

    ri_sell_levels: dict[int, int] = {}
    ri_buy_levels : dict[int, int] = {}

    for h in active_hours:
        qty = lp_qty.get(h, 0.0)
        if qty > 1e-9 and max_dis_step > 1e-9:
            ri_sell_levels[h] = min(n_levels, math.ceil(qty / max_dis_step * n_levels))
        else:
            ri_sell_levels[h] = 0
        if qty < -1e-9 and max_chg_step > 1e-9:
            ri_buy_levels[h] = min(n_levels, math.ceil(abs(qty) / max_chg_step * n_levels))
        else:
            ri_buy_levels[h] = 0

    return ri_sell_levels, ri_buy_levels


# ── Regime computation ────────────────────────────────────────────────────────

def compute_regimes(auction_mids: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute per-hour regime parameters from 24-hour D-1 auction MID prices.

    Returns dict with per-hour arrays of shape (24,):
        p_min        : p_min,k for the buy regime containing each hour
        p_max        : p_max,k for the sell regime containing each hour
        T_s          : hours from session start to delivery of buy-regime minimum
        T_b          : hours from session start to delivery of sell-regime maximum
        auction_mid  : original per-hour mids (for α₅ RI reference)
    """
    n = len(auction_mids)
    assert n == 24, f"Expected 24-hour auction curve, got length {n}"

    # find the local minima and maxima on the auction curve 
    def _segment_boundaries(arr: np.ndarray, mode: str) -> list[int]:
        result = [0]
        for i in range(1, len(arr) - 1):
            if mode == "max" and arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
                result.append(i)
            elif mode == "min" and arr[i] <= arr[i - 1] and arr[i] <= arr[i + 1]:
                result.append(i)
        if result[-1] != len(arr) - 1:
            result.append(len(arr) - 1)
        return result

    p_min_arr = np.empty(n)
    T_s_arr   = np.empty(n)
    p_max_arr = np.empty(n)
    T_b_arr   = np.empty(n)

    for start, end in zip(*_make_pairs(_segment_boundaries(auction_mids, "max"))):
        seg       = auction_mids[start : end + 1]
        local_min = start + int(np.argmin(seg))
        p_min_arr[start : end + 1] = auction_mids[local_min]
        T_s_arr[start : end + 1]   = float(local_min) + 9.0

    for start, end in zip(*_make_pairs(_segment_boundaries(auction_mids, "min"))):
        seg       = auction_mids[start : end + 1]
        local_max = start + int(np.argmax(seg))
        p_max_arr[start : end + 1] = auction_mids[local_max]
        T_b_arr[start : end + 1]   = float(local_max) + 9.0

    return {
        "p_min"      : p_min_arr,
        "p_max"      : p_max_arr,
        "T_s"        : T_s_arr,
        "T_b"        : T_b_arr,
        "auction_mid": auction_mids.copy(),
    }


def _make_pairs(boundaries: list[int]) -> tuple[list[int], list[int]]:
    return boundaries[:-1], boundaries[1:]


# ── Policy ────────────────────────────────────────────────────────────────────

class AlphaPolicy(nn.Module):
    """
    B&P threshold policy for multi-hour daily episodes.

    n_levels: number of non-zero quantity levels per side (default 1 → binary).
              With n_levels=1: actions are {0, max} per side (sell OR buy).
              With n_levels=k: actions are {0, max/k, 2*max/k, …, max}.

    In our synthetic data all per-step quantities fall within the best bid/ask
    level, so n_levels > 1 does not change effective prices (degenerate). The
    general CDF formulation is kept for correctness with real-depth data.
    """

    def __init__(
        self,
        alpha_s1_init  : float = 0.5,
        alpha_b1_init  : float = 0.5,
        alpha_s2_init  : float = 0.0,
        alpha_b2_init  : float = 0.0,
        alpha_s3_init  : float = 0.0,
        alpha_b3_init  : float = 0.0,
        alpha_s4_init  : float = 1.0,
        alpha_b4_init  : float = 1.0,
        alpha_s5_init  : float = 0.0,
        alpha_b5_init  : float = 0.0,
        log_sigma_init : float = 1.6,
        round_trip_eff : float = 1.0,
        n_levels       : int   = 1,
    ) -> None:
        super().__init__()
        self.alpha_s1    = nn.Parameter(torch.tensor(alpha_s1_init))
        self.alpha_b1    = nn.Parameter(torch.tensor(alpha_b1_init))
        self.alpha_s2    = nn.Parameter(torch.tensor(alpha_s2_init))
        self.alpha_b2    = nn.Parameter(torch.tensor(alpha_b2_init))
        self.alpha_s3    = nn.Parameter(torch.tensor(alpha_s3_init))
        self.alpha_b3    = nn.Parameter(torch.tensor(alpha_b3_init))
        self.alpha_s4    = nn.Parameter(torch.tensor(alpha_s4_init))
        self.alpha_b4    = nn.Parameter(torch.tensor(alpha_b4_init))
        self.alpha_s5    = nn.Parameter(torch.tensor(alpha_s5_init))
        self.alpha_b5    = nn.Parameter(torch.tensor(alpha_b5_init))
        self.log_sigma_X = nn.Parameter(torch.tensor(log_sigma_init))
        self.log_sigma_Y = nn.Parameter(torch.tensor(log_sigma_init))

        self._eta         = float(round_trip_eff ** 0.5)
        self.n_levels     = n_levels
        self._regime: dict[str, np.ndarray] | None = None
        self._capacity    : float = 1.0
        self._initial_soc : float = 0.0
        self._max_charge  : float = 1.0
        self._max_dis     : float = 1.0

    def set_regime_info(self, regime: dict[str, np.ndarray]) -> None:
        self._regime = regime

    def set_env_params(
        self,
        capacity    : float,
        initial_soc : float,
        max_charge  : float,
        max_dis     : float,
    ) -> None:
        self._capacity    = capacity
        self._initial_soc = initial_soc
        self._max_charge  = max_charge
        self._max_dis     = max_dis

    # ── §V-F imbalance prevention ────────────────────────────────────────────

    def _per_hour_feasibility(
        self,
        active_hours: list[int],
        position    : np.ndarray,
        step_hours  : float,
    ) -> tuple[dict[int, int], dict[int, int]]:
        """
        Per-hour maximum feasible sell/buy levels given current SOC trajectory.

        Uses the same SOC look-ahead as the environment (§V-F): available
        discharge = min SOC over remaining hours, headroom = max SOC over
        remaining hours.  Infeasible levels are zeroed out by callers.
        """
        soc = self._initial_soc - np.cumsum(position)

        sell_max: dict[int, int] = {}
        buy_max:  dict[int, int] = {}

        sell_step = self._max_dis    * step_hours   # global step max (for 1 level)
        buy_step  = self._max_charge * step_hours
        sell_unit = sell_step / self.n_levels if self.n_levels > 0 else sell_step
        buy_unit  = buy_step  / self.n_levels if self.n_levels > 0 else buy_step

        for h in active_hours:
            available = float(max(0.0, soc[h:].min()))
            feasible_sell_qty = min(sell_step, available)
            sell_max[h] = int(min(self.n_levels, feasible_sell_qty / sell_unit)) \
                          if sell_unit > 1e-9 else 0

            headroom = float(max(0.0, self._capacity - soc[h:].max()))
            feasible_buy_qty = min(buy_step, headroom)
            buy_max[h] = int(min(self.n_levels, feasible_buy_qty / buy_unit)) \
                         if buy_unit > 1e-9 else 0

        return sell_max, buy_max


    # ── Vectorised threshold means over all active hours ─────────────────────

    def _compute_threshold_means(
        self,
        active_hours  : list[int],
        best_bids     : dict[int, float],
        best_asks     : dict[int, float],
        order_books   : dict[int, dict],
        elapsed_hours : float,
        v_end_fraction: float,
        position      : np.ndarray | None,
        step_hours    : float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int], dict[int, int]]:
        """
        Returns (mu_X, mu_Y, sigma_X, sigma_Y, ri_sell_levels, ri_buy_levels).

        ri_sell_levels[h] / ri_buy_levels[h]: integer in {0, n_levels} indicating
        the quantity level RI accepts.  Callers apply Step 4 via auxiliary Gaussian
        probability reallocation (§V-E, Appendix D) — NOT a mean shift on mu.
        """
        n   = len(active_hours)
        h_a = np.array(active_hours)

        p_min = self._regime["p_min"][h_a]
        p_max = self._regime["p_max"][h_a]
        T_s   = self._regime["T_s"][h_a]
        T_b   = self._regime["T_b"][h_a]
        mids  = self._regime["auction_mid"][h_a]

        bids = np.array([best_bids[h] for h in active_hours], dtype=np.float32)
        asks = np.array([best_asks[h] for h in active_hours], dtype=np.float32)

        price_range = np.maximum(p_max - p_min, 1e-3).astype(np.float32)
        pr = torch.from_numpy(price_range)
        pm = torch.from_numpy(p_min.astype(np.float32))
        pM = torch.from_numpy(p_max.astype(np.float32))
        ds = torch.from_numpy((elapsed_hours - T_s).astype(np.float32))
        db = torch.from_numpy((elapsed_hours - T_b).astype(np.float32))

        # Compute RI signals via LP (§V-E, Appendix E); returns integer levels
        if position is not None:
            ri_sell_levels, ri_buy_levels = _rolling_intrinsic_lp_levels(
                active_hours, best_bids, best_asks, position,
                self._capacity, self._initial_soc,
                self._max_charge, self._max_dis, self._eta, step_hours,
                self.n_levels,
                order_books=order_books,
            )
        else:
            ri_sell_levels = {h: (self.n_levels if float(bids[i]) > float(mids[i]) else 0)
                              for i, h in enumerate(active_hours)}
            ri_buy_levels  = {h: (self.n_levels if float(asks[i]) < float(mids[i]) else 0)
                              for i, h in enumerate(active_hours)}

        # Step 1 — auction anchor
        mu_X = self.alpha_s1 * pr + pm
        mu_Y = pM - self.alpha_b1 * pr
        # Step 2 — position adjustment
        mu_X = mu_X - self.alpha_s2 * v_end_fraction
        mu_Y = mu_Y - self.alpha_b2 * v_end_fraction
        # Step 3 — urgency
        half = pr / 2.0
        mu_X = mu_X + self.alpha_s3 * half * torch.exp((self.alpha_s4 * ds).clamp(max=5.0))
        mu_Y = mu_Y - self.alpha_b3 * half * torch.exp((self.alpha_b4 * db).clamp(max=5.0))
        # Step 4 — RI reference applied as auxiliary Gaussian in callers (not a mean shift)
        # Step 5 — §V-G efficiency adjustment (no new parameters).
        # Main (financial) distribution uses η·µ_Y and µ_X/η (always, for all hours).
        # Auxiliary (physical) distribution uses µ_Y/η² and η²·µ_X.
        # When efficiency=1 (η=1): main = physical = µ, reducing to plain §V-E.
        eta    = self._eta          # one-way efficiency √(round-trip)
        rt_eff = eta * eta          # round-trip efficiency

        mu_X_phys = mu_X * rt_eff  # η_in · η_out · µ_X  (physical buy auxiliary)
        mu_Y_phys = mu_Y / rt_eff  # µ_Y / (η_in · η_out) (physical sell auxiliary)
        mu_X      = mu_X / eta     # µ_X / η_out           (financial buy baseline)
        mu_Y      = mu_Y * eta     # η_in · µ_Y            (financial sell baseline)

        sigma_X = torch.exp(self.log_sigma_X).clamp(min=1e-3).expand(n)
        sigma_Y = torch.exp(self.log_sigma_Y).clamp(min=1e-3).expand(n)

        return mu_X, mu_Y, mu_X_phys, mu_Y_phys, sigma_X, sigma_Y, ri_sell_levels, ri_buy_levels

    # ── Batched stochastic forward (training) ────────────────────────────────

    def forward_batch(
        self,
        active_hours  : list[int],
        best_bids     : dict[int, float],
        best_asks     : dict[int, float],
        order_books   : dict[int, dict],
        elapsed_hours : float,
        v_end_fraction: float,
        position      : np.ndarray | None = None,
        step_hours    : float = _STEP_HOURS,
    ) -> tuple[dict[int, float], torch.Tensor]:
        """
        Sample discrete quantity actions for all active hours.

        Returns:
            actions       : {delivery_hour → signed quantity MWh}
                            positive = sell/discharge, negative = buy/charge
            joint_log_prob: Σ_h log P(a_h | s) for REINFORCE
        """
        if not active_hours:
            return {}, torch.zeros(())

        if self._regime is None:
            raise RuntimeError("Call set_regime_info() before using the policy.")

        mu_X, mu_Y, mu_X_phys, mu_Y_phys, sigma_X, sigma_Y, ri_sell_levels, ri_buy_levels = (
            self._compute_threshold_means(
                active_hours, best_bids, best_asks, order_books,
                elapsed_hours, v_end_fraction, position, step_hours,
            )
        )

        max_sell = self._max_dis    * step_hours
        max_buy  = self._max_charge * step_hours

        # Per-hour feasibility caps (§V-F): imbalance prevention
        if position is not None:
            sell_feasible, buy_feasible = self._per_hour_feasibility(
                active_hours, position, step_hours
            )
        else:
            sell_feasible = {h: self.n_levels for h in active_hours}
            buy_feasible  = {h: self.n_levels for h in active_hours}

        actions: dict[int, float] = {}
        joint_log_prob = torch.zeros(())

        for i, h in enumerate(active_hours):
            book = order_books.get(h, {"bids": [], "asks": []})

            p_sell_t = torch.tensor(
                sell_midpoint_prices(book["bids"], self.n_levels, max_sell), dtype=torch.float32
            )
            p_buy_t  = torch.tensor(
                buy_midpoint_prices(book["asks"],  self.n_levels, max_buy),  dtype=torch.float32
            )
            # Financial (main) distributions — §V-G baseline
            dist_Y_i      = Normal(mu_Y[i],      sigma_Y[i])
            dist_X_i      = Normal(mu_X[i],      sigma_X[i])
            # Physical (auxiliary base) distributions — §V-G, used for non-RI levels
            dist_Y_phys_i = Normal(mu_Y_phys[i], sigma_Y[i])
            dist_X_phys_i = Normal(mu_X_phys[i], sigma_X[i])

            # RI + efficiency auxiliary Gaussian reallocation (§V-E + §V-G, Appendix D)
            p_sell = _sell_level_probs_ri(
                p_sell_t, dist_Y_i, dist_Y_phys_i, self.alpha_b5, ri_sell_levels[h], self.n_levels
            )
            p_buy  = _buy_level_probs_ri(
                p_buy_t,  dist_X_i, dist_X_phys_i, self.alpha_s5, ri_buy_levels[h],  self.n_levels
            )

            # Imbalance prevention: redirect mass from infeasible levels (§V-F)
            p_sell = _reweight_feasibility(p_sell, sell_feasible[h])
            p_buy  = _reweight_feasibility(p_buy,  buy_feasible[h])

            # Sample discrete levels from categorical (detached — gradient via log_prob below)
            sell_level = int(Categorical(p_sell.detach()).sample().item())
            buy_level  = int(Categorical(p_buy.detach()).sample().item()) if sell_level == 0 else 0

            # Signed quantity (sell priority)
            if sell_level > 0:
                actions[h] = sell_level * max_sell / self.n_levels
            elif buy_level > 0:
                actions[h] = -buy_level * max_buy / self.n_levels
            else:
                actions[h] = 0.0

            # Log probability: P(sell k) or P(sell=0)×P(buy k/0) (§IV-A, Appendix C)
            if sell_level > 0:
                lp = torch.log(p_sell[sell_level])
            elif buy_level > 0:
                lp = torch.log(p_sell[0]) + torch.log(p_buy[buy_level])
            else:
                lp = torch.log(p_sell[0]) + torch.log(p_buy[0])

            joint_log_prob = joint_log_prob + lp

        return actions, joint_log_prob

    # ── Batched greedy forward (evaluation) ──────────────────────────────────

    @torch.no_grad()
    def greedy_batch(
        self,
        active_hours  : list[int],
        best_bids     : dict[int, float],
        best_asks     : dict[int, float],
        order_books   : dict[int, dict],
        elapsed_hours : float,
        v_end_fraction: float,
        position      : np.ndarray | None = None,
        step_hours    : float = _STEP_HOURS,
        fast          : bool  = False,
    ) -> dict[int, float]:
        """
        Deterministic actions by comparing threshold means directly against
        midpoint prices (no sampling). For evaluation only.

        fast=True: skips the RI LP and uses the auction-mid RI reference instead
        (position=None path in _compute_threshold_means). Feasibility checking
        still uses the actual position. Typical speedup: 10-100× per tick.
        fast=False (default): full LP-based RI, matching training behaviour.
        """
        if not active_hours:
            return {}

        if self._regime is None:
            raise RuntimeError("Call set_regime_info() before using the policy.")

        # When fast=True, skip LP by passing position=None to _compute_threshold_means.
        # µ_X and µ_Y (Steps 1-4) are unaffected; only ri_sell/buy_levels use the LP.
        lp_position = None if fast else position

        mu_X, mu_Y, mu_X_phys, mu_Y_phys, sigma_X, sigma_Y, ri_sell_levels, ri_buy_levels = (
            self._compute_threshold_means(
                active_hours, best_bids, best_asks, order_books,
                elapsed_hours, v_end_fraction, lp_position, step_hours,
            )
        )

        max_sell = self._max_dis    * step_hours
        max_buy  = self._max_charge * step_hours

        # Feasibility always uses the actual position (no LP needed)
        if position is not None:
            sell_feasible, buy_feasible = self._per_hour_feasibility(
                active_hours, position, step_hours
            )
        else:
            sell_feasible = {h: self.n_levels for h in active_hours}
            buy_feasible  = {h: self.n_levels for h in active_hours}

        actions: dict[int, float] = {}
        for i, h in enumerate(active_hours):
            book = order_books.get(h, {"bids": [], "asks": []})

            p_sell_t = torch.tensor(
                sell_midpoint_prices(book["bids"], self.n_levels, max_sell), dtype=torch.float32
            )
            p_buy_t  = torch.tensor(
                buy_midpoint_prices(book["asks"],  self.n_levels, max_buy),  dtype=torch.float32
            )
            dist_Y_i      = Normal(mu_Y[i],      sigma_Y[i])
            dist_X_i      = Normal(mu_X[i],      sigma_X[i])
            dist_Y_phys_i = Normal(mu_Y_phys[i], sigma_Y[i])
            dist_X_phys_i = Normal(mu_X_phys[i], sigma_X[i])

            p_sell = _sell_level_probs_ri(
                p_sell_t, dist_Y_i, dist_Y_phys_i, self.alpha_b5, ri_sell_levels[h], self.n_levels
            )
            p_buy  = _buy_level_probs_ri(
                p_buy_t,  dist_X_i, dist_X_phys_i, self.alpha_s5, ri_buy_levels[h],  self.n_levels
            )

            # Imbalance prevention: redirect mass from infeasible levels (§V-F)
            p_sell = _reweight_feasibility(p_sell, sell_feasible[h])
            p_buy  = _reweight_feasibility(p_buy,  buy_feasible[h])

            # Greedy: mode of the feasibility-adjusted distribution
            sell_level = int(p_sell.argmax().item())
            buy_level  = int(p_buy.argmax().item()) if sell_level == 0 else 0

            if sell_level > 0:
                actions[h] = sell_level * max_sell / self.n_levels
            elif buy_level > 0:
                actions[h] = -buy_level * max_buy / self.n_levels
            else:
                actions[h] = 0.0

        return actions

    @torch.no_grad()
    def threshold_means_batch(
        self,
        active_hours  : list[int],
        best_bids     : dict[int, float],
        best_asks     : dict[int, float],
        elapsed_hours : float,
        v_end_fraction: float,
    ) -> dict[int, dict[str, float]]:
        """
        Financial threshold means µ_X (buy) and µ_Y (sell) per active hour.

        Diagnostic method for evaluation — does not run the RI LP (position=None
        uses the fast auction-mid fallback), so it adds zero overhead on top of
        a greedy_batch call.  µ_X / µ_Y are Steps 1-4 outputs (efficiency-adjusted
        to the financial baseline); they are unaffected by the RI LP skip.
        """
        if not active_hours or self._regime is None:
            return {}
        mu_X, mu_Y, *_ = self._compute_threshold_means(
            active_hours, best_bids, best_asks, {},
            elapsed_hours, v_end_fraction,
            position=None,      # skips LP — µ_X/µ_Y are unaffected
            step_hours=_STEP_HOURS,
        )
        return {
            h: {"mu_X": float(mu_X[i].item()), "mu_Y": float(mu_Y[i].item())}
            for i, h in enumerate(active_hours)
        }

    def param_snapshot(self) -> dict[str, float]:
        return {
            "alpha_s1": self.alpha_s1.item(),
            "alpha_b1": self.alpha_b1.item(),
            "alpha_s2": self.alpha_s2.item(),
            "alpha_b2": self.alpha_b2.item(),
            "alpha_s3": self.alpha_s3.item(),
            "alpha_b3": self.alpha_b3.item(),
            "alpha_s4": self.alpha_s4.item(),
            "alpha_b4": self.alpha_b4.item(),
            "alpha_s5": self.alpha_s5.item(),
            "alpha_b5": self.alpha_b5.item(),
            "sigma_X" : torch.exp(self.log_sigma_X).item(),
            "sigma_Y" : torch.exp(self.log_sigma_Y).item(),
        }
