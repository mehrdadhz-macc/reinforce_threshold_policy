"""
Multi-hour market environment for one full CIM trading day.

One episode = all 24 delivery hours of a Berlin calendar day.
One step    = one CIM tick (1 second for paper-faithful data; 1 minute for synthetic).

At each tick, the agent trades ALL active delivery hours simultaneously
with one independent signed-quantity action per active delivery hour.

Action encoding:
    action[d] > 0  → discharge / sell  action[d] MWh for delivery hour d
    action[d] < 0  → charge   / buy   |action[d]| MWh for delivery hour d
    action[d] = 0  → idle

Position model:
    position[d] = net MWh committed for delivery hour d (cumulative over episode).
                  positive → net discharge (will deliver energy at delivery)
                  negative → net charge    (will receive  energy at delivery)

    implied_soc[d] = initial_soc − cumsum(position)[d]
    Constraint: 0 ≤ implied_soc[d] ≤ capacity  for all d

Reward (integral under the order book curve, paper §III-C):
    SELL q MWh at hour d:
        revenue = ∫₀^q  p_bid(z, d) dz          (no efficiency factor — §III-C)
        position[d] += q
    BUY  q MWh at hour d:
        cost    = ∫₀^q  p_ask(z, d) dz           (no efficiency factor — §III-C)
        position[d] -= q
    IDLE: reward = 0

    Efficiency is accounted for entirely through the policy parameterization (§V-G),
    not the reward signal.  The RI LP benchmark retains efficiency in its objective
    (Appendix E, Eq. 25) since it is a separate physical optimisation.

    In practice, since per-step quantities (~0.83 MWh at 1-min) are much smaller
    than individual order sizes (~5–20 MWh), the integral reduces to best_price × q.
    The integral formulation is kept for correctness and future data compatibility.

    Curriculum training (§VI-A):
    With tick_stride > 1, each visible tick spans stride minutes; the per-step
    quantity caps scale accordingly (step_hours = stride / 60).  The environment
    exposes step_hours in StepState so the policy uses the correct scale.

Terminal:
    terminal_penalty × max(0, implied_soc[23])  — discourages leftover energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_STEP_HOURS = 1 / 60
N_HOURS = 24

# Order book level: (price_eur_mwh, quantity_mwh)
BookLevel = tuple[float, float]


@dataclass
class StepState:
    active_hours  : list[int]              # delivery hours 0–23 still open
    best_bid      : dict[int, float]       # best bid price EUR/MWh per active hour
    best_ask      : dict[int, float]       # best ask price EUR/MWh per active hour
    # Full order book depth per active hour.
    # order_book[h]["bids"]: [(price, qty), ...] sorted price DESC (best bid first)
    # order_book[h]["asks"]: [(price, qty), ...] sorted price ASC  (best ask first)
    order_book    : dict[int, dict[str, list[BookLevel]]]
    position      : np.ndarray             # shape (24,) net MWh per hour (+ = discharge)
    v_end_fraction: float                  # implied_soc[23] / capacity
    elapsed_hours : float                  # hours since session start (15:00 D-1 UTC)
    step_hours    : float = _STEP_HOURS    # trading window this tick (stride / 60)


def fill_order(book: list[BookLevel], quantity: float) -> tuple[float, float]:
    """
    Walk an order book to fill `quantity` MWh.

    For sell execution, pass the bids book (price DESC).
    For buy  execution, pass the asks book (price ASC).

    Returns:
        (volume_filled, total_value)
        volume_filled ≤ quantity if the book is too shallow.
        total_value is the sum of price × partial_qty across levels.
    """
    remaining = quantity
    total_value = 0.0
    for price, level_qty in book:
        if remaining <= 0.0:
            break
        fill = min(remaining, level_qty)
        total_value += price * fill
        remaining -= fill
    volume_filled = quantity - max(0.0, remaining)
    return volume_filled, total_value


class MultiHourMarketEnv:
    """Episode = one Berlin calendar day; step = one CIM tick (second- or minute-level)."""

    def __init__(
        self,
        capacity_mwh             : float = 100.0,
        max_charge_mw            : float = 25.0,
        max_discharge_mw         : float = 25.0,
        efficiency               : float = 0.95,
        initial_soc_mwh          : float = 0.0,
        terminal_penalty_eur_mwh : float = 30.0,
    ) -> None:
        self.capacity         = capacity_mwh
        self.max_charge       = max_charge_mw
        self.max_discharge    = max_discharge_mw
        self.one_way_eff      = efficiency ** 0.5
        self.initial_soc      = initial_soc_mwh
        self.terminal_penalty = terminal_penalty_eur_mwh

        self._ticks              : list[pd.Timestamp] = []
        self._tick_data          : dict               = {}
        self._last_tick_per_hour : dict[int, int]     = {}
        self._position           = np.zeros(N_HOURS)
        self._tick_idx           = 0
        self._n_ticks            = 0
        self._session_start      : pd.Timestamp | None = None
        self._step_hours         : float = _STEP_HOURS

    # ── Setup ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        cim_day        : pd.DataFrame,
        delivery_starts: list[pd.Timestamp],
        session_start  : pd.Timestamp,
        tick_stride    : int = 1,
    ) -> StepState:
        """
        Begin a new day episode.

        Args:
            cim_day         : all CIM rows for the 24 delivery hours of this day.
            delivery_starts : list of 24 UTC delivery_start timestamps (hour 0 first).
            session_start   : 15:00 D-1 expressed in UTC.
            tick_stride     : subsample every Nth minute tick for curriculum training.
        """
        assert len(delivery_starts) == N_HOURS
        self._session_start = session_start
        self._position      = np.zeros(N_HOURS)
        self._tick_idx      = 0

        self._tick_data, all_ticks, self._last_tick_per_hour = (
            self._build_tick_data(cim_day, delivery_starts)
        )
        # Compute step_hours from actual tick spacing so the environment works
        # correctly for both minute-level (freq="min") and second-level (freq="s") data.
        if len(all_ticks) >= 2:
            delta_ns = all_ticks[1].value - all_ticks[0].value
            base_step_hours = delta_ns / 1e9 / 3600.0
        else:
            base_step_hours = _STEP_HOURS
        self._step_hours = tick_stride * base_step_hours

        if tick_stride > 1:
            indices  = list(range(0, len(all_ticks), tick_stride))
            last_idx = len(all_ticks) - 1
            if last_idx not in indices:
                indices.append(last_idx)
            self._ticks = [all_ticks[i] for i in sorted(indices)]
        else:
            self._ticks = all_ticks
        self._n_ticks = len(self._ticks)
        return self._state()

    # ── Per-step interface ────────────────────────────────────────────────────

    def step(
        self,
        actions: dict[int, float],
    ) -> tuple[StepState, float, bool, dict]:
        """
        Apply per-hour signed-quantity actions at the current tick.

        Args:
            actions : {delivery_hour_index → signed_quantity_mwh}
                      positive = discharge/sell, negative = charge/buy, 0 = idle.
        Returns:
            (next_state, total_reward, done, info)
        """
        ts         = self._ticks[self._tick_idx]
        tick_entry = self._tick_data.get(ts, {})

        total_reward = 0.0
        info: dict[int, dict] = {}

        for hour, qty in actions.items():
            if hour not in tick_entry or qty == 0.0:
                info[hour] = {"qty": 0.0, "revenue": 0.0}
                continue

            book = tick_entry[hour]

            if qty > 0.0:
                # Sell / discharge — paper §III-C: rev = ∫ p(z) dz, no efficiency factor
                max_q = self._max_discharge(hour)
                q     = min(qty, max_q)
                if q > 0.0:
                    vol, value    = fill_order(book["bids"], q)
                    self._position[hour] += vol
                    total_reward         += value
                    info[hour] = {"qty": vol, "revenue": value}
                else:
                    info[hour] = {"qty": 0.0, "revenue": 0.0}

            else:
                # Buy / charge — paper §III-C: cost = ∫ p(z) dz, no efficiency factor
                q_req = abs(qty)
                max_q = self._max_charge(hour)
                q     = min(q_req, max_q)
                if q > 0.0:
                    vol, value    = fill_order(book["asks"], q)
                    self._position[hour] -= vol
                    total_reward         -= value
                    info[hour] = {"qty": -vol, "revenue": -value}
                else:
                    info[hour] = {"qty": 0.0, "revenue": 0.0}

        self._tick_idx += 1
        done = self._tick_idx >= self._n_ticks

        if done:
            leftover = max(0.0, self._implied_soc()[N_HOURS - 1])
            total_reward -= self.terminal_penalty * leftover

        next_state = self._state() if not done else self._terminal_state()
        return next_state, total_reward, done, info

    @property
    def current_prices(self) -> dict[int, dict[str, float]]:
        """Best bid/ask per active delivery hour at the current tick."""
        if self._tick_idx >= self._n_ticks:
            return {}
        ts = self._ticks[self._tick_idx]
        return {
            h: {
                "best_bid": entry["bids"][0][0] if entry["bids"] else 0.0,
                "best_ask": entry["asks"][0][0] if entry["asks"] else float("inf"),
            }
            for h, entry in self._tick_data.get(ts, {}).items()
        }

    def implied_soc(self) -> np.ndarray:
        """Public accessor: implied SoC trajectory across 24 hours."""
        return self._implied_soc()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _implied_soc(self) -> np.ndarray:
        # position[d] > 0 means net discharge committed → SOC decreases
        return self.initial_soc - np.cumsum(self._position)

    def _max_charge(self, hour: int) -> float:
        soc      = self._implied_soc()
        headroom = self.capacity - soc[hour:].max()
        by_power = self.max_charge * self._step_hours
        return max(0.0, min(by_power, headroom))

    def _max_discharge(self, hour: int) -> float:
        soc       = self._implied_soc()
        available = soc[hour:].min()
        by_power  = self.max_discharge * self._step_hours
        return max(0.0, min(by_power, available))

    def _state(self) -> StepState:
        ts         = self._ticks[self._tick_idx]
        tick_entry = self._tick_data.get(ts, {})
        active     = sorted(tick_entry.keys())

        elapsed_h = (ts.value - self._session_start.value) / 1e9 / 3600

        soc   = self._implied_soc()
        v_end = soc[N_HOURS - 1] / self.capacity if self.capacity > 0 else 0.0

        best_bid = {h: tick_entry[h]["bids"][0][0] if tick_entry[h]["bids"] else 0.0
                    for h in active}
        best_ask = {h: tick_entry[h]["asks"][0][0] if tick_entry[h]["asks"] else float("inf")
                    for h in active}

        return StepState(
            active_hours   = active,
            best_bid       = best_bid,
            best_ask       = best_ask,
            order_book     = {h: tick_entry[h] for h in active},
            position       = self._position.copy(),
            v_end_fraction = float(v_end),
            elapsed_hours  = float(elapsed_h),
            step_hours     = self._step_hours,
        )

    def _terminal_state(self) -> StepState:
        soc   = self._implied_soc()
        v_end = soc[N_HOURS - 1] / self.capacity if self.capacity > 0 else 0.0
        return StepState(
            active_hours   = [],
            best_bid       = {},
            best_ask       = {},
            order_book     = {},
            position       = self._position.copy(),
            v_end_fraction = float(v_end),
            elapsed_hours  = float("inf"),
            step_hours     = self._step_hours,
        )

    def _build_tick_data(
        self,
        cim            : pd.DataFrame,
        delivery_starts: list[pd.Timestamp],
    ) -> tuple[dict, list, dict]:
        """
        Build per-tick order book lookup.

        Returns:
            tick_data : {ts → {hour → {"bids": [(price, qty)…], "asks": [(price, qty)…]}}}
            ticks     : sorted list of timestamps
            last_tick_per_hour : {hour → last tick index}

        Uses vectorised groupby aggregation (28× faster than Python for-loop over groups).
        """
        delivery_map = {ds: h for h, ds in enumerate(delivery_starts)}

        cim_mapped = cim.copy()
        cim_mapped["hour"] = cim_mapped["delivery_start"].map(delivery_map)
        cim_mapped = cim_mapped.dropna(subset=["hour"])
        cim_mapped["hour"] = cim_mapped["hour"].astype(int)

        # Vectorised: sort once per side, then groupby-agg into lists.
        bid_df = (
            cim_mapped[cim_mapped["side"] == "buy"]
            .sort_values(["timestamp", "hour", "price_eur_mwh"],
                         ascending=[True, True, False])
        )
        ask_df = (
            cim_mapped[cim_mapped["side"] == "sell"]
            .sort_values(["timestamp", "hour", "price_eur_mwh"],
                         ascending=[True, True, True])
        )

        bid_agg = bid_df.groupby(["timestamp", "hour"])[
            ["price_eur_mwh", "quantity_mwh"]
        ].agg(list)
        ask_agg = ask_df.groupby(["timestamp", "hour"])[
            ["price_eur_mwh", "quantity_mwh"]
        ].agg(list)

        # Build flat dicts: (ts, hour) → [(price, qty), …]
        bid_book: dict[tuple, list] = {
            (ts, h): list(zip(p, q))
            for (ts, h), (p, q) in zip(bid_agg.index, bid_agg.values)
        }
        ask_book: dict[tuple, list] = {
            (ts, h): list(zip(p, q))
            for (ts, h), (p, q) in zip(ask_agg.index, ask_agg.values)
        }

        # Merge into nested tick_data dict
        all_keys = set(bid_book) | set(ask_book)
        tick_data: dict = {}
        for ts, hour in all_keys:
            if ts not in tick_data:
                tick_data[ts] = {}
            tick_data[ts][hour] = {
                "bids": bid_book.get((ts, hour), []),
                "asks": ask_book.get((ts, hour), []),
            }

        ticks        = sorted(tick_data.keys())
        tick_idx_map = {ts: i for i, ts in enumerate(ticks)}

        last_tick_per_hour: dict[int, int] = {}
        for ts, hour_dict in tick_data.items():
            idx = tick_idx_map[ts]
            for h in hour_dict:
                if h not in last_tick_per_hour or idx > last_tick_per_hour[h]:
                    last_tick_per_hour[h] = idx

        return tick_data, ticks, last_tick_per_hour
