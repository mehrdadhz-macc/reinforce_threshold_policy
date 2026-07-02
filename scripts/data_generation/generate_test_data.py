"""
Generate a synthetic test dataset with the same market structure as the
training data but a different random seed and date range.

Training data : seed=42,  2023-01-01 – 2023-01-30 (CET, no DST transition)
Test data     : seed=123, 2023-04-01 – 2023-04-02 (CEST, no DST transition)

TICK_FREQ = "s" matches the paper's 1-second EPEX SPOT evaluation data (§VI-A).
N_DAYS is kept small (2 days) to keep the generated file to ~1-2 GB.
Increase N_DAYS for more comprehensive evaluation.

March is intentionally avoided: the spring-forward DST on 2023-03-26 causes
`delivery_day.replace(hour=2)` to silently fold into hour=3, merging two
independently-drawn order books into one group and violating the spread
constraint. April is entirely in CEST (UTC+2) with no such ambiguity.

Run from the project root:
    venv/bin/python3 scripts/data_generation/generate_test_data.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

SEED       = 123
TIMEZONE   = "Europe/Berlin"
N_DAYS     = 31   # 1 month: August 2023 (held-out, after training period)
START_DATE = "2023-08-01"

# Tick frequency: "s" = 1-second (paper §VI-A), "min" = 1-minute (practical default)
# 30 days at "s" resolution requires ~50 GB; "min" is used for synthetic data.
TICK_FREQ  = "min"

N_CIM_LEVELS     = 5
N_AUCTION_LEVELS = 10

OUT_DIR      = Path(__file__).parent.parent.parent / "data" / "test"
CIM_FILE     = OUT_DIR / "cim_order_book.csv"
AUCTION_FILE = OUT_DIR / "intraday_auction_curves.csv"

# Hourly base price shape identical to training data (EUR/MWh, hours 0-23)
HOURLY_BASE_PRICE = np.array([
    35, 33, 32, 31, 32, 35,
    42, 52, 58, 56, 54, 53,
    52, 51, 52, 54, 58, 62,
    65, 63, 58, 52, 45, 40,
], dtype=float)


# ── Generation ────────────────────────────────────────────────────────────────

def generate_auction_curves(delivery_days: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for delivery_day in delivery_days:
        auction_time = (delivery_day - pd.Timedelta(days=1)).replace(
            hour=15, minute=0, second=0, microsecond=0
        )
        is_weekend = delivery_day.dayofweek >= 5

        for hour in range(24):
            delivery_start = delivery_day.replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
            base = HOURLY_BASE_PRICE[hour] * (0.85 if is_weekend else 1.0)
            mid  = base + np.random.normal(0, 3.0)
            half_spread = np.random.uniform(1.0, 3.0)

            sell_steps  = np.cumsum(np.random.uniform(0.5, 2.0, N_AUCTION_LEVELS))
            sell_prices = (mid + half_spread + sell_steps).round(2)
            sell_qtys   = np.random.uniform(1.0, 50.0, N_AUCTION_LEVELS).round(2)

            buy_steps  = np.cumsum(np.random.uniform(0.5, 2.0, N_AUCTION_LEVELS))
            buy_prices = (mid - half_spread - buy_steps).round(2)
            buy_qtys   = np.random.uniform(1.0, 50.0, N_AUCTION_LEVELS).round(2)

            for lvl in range(N_AUCTION_LEVELS):
                rows.append((auction_time, delivery_start,
                             "sell", sell_prices[lvl], sell_qtys[lvl], lvl + 1))
                rows.append((auction_time, delivery_start,
                             "buy",  buy_prices[lvl],  buy_qtys[lvl],  lvl + 1))

    return pd.DataFrame(rows, columns=[
        "auction_time", "delivery_start", "side",
        "price_eur_mwh", "quantity_mwh", "level",
    ])


def generate_cim_order_book(delivery_days: pd.DatetimeIndex) -> pd.DataFrame:
    dfs = []
    for delivery_day in delivery_days:
        trading_open = (delivery_day - pd.Timedelta(days=1)).replace(
            hour=15, minute=0, second=0, microsecond=0
        )
        is_weekend = delivery_day.dayofweek >= 5

        for hour in range(24):
            delivery_start = delivery_day.replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
            delivery_end  = delivery_start + pd.Timedelta(hours=1)
            trading_close = delivery_start - pd.Timedelta(minutes=30)

            ticks = pd.date_range(
                start=trading_open,
                end=trading_close,
                freq=TICK_FREQ,
                inclusive="left",
            )
            n_ticks = len(ticks)
            if n_ticks == 0:
                continue

            # Brownian-motion scaling: per-minute variance = 0.01 regardless of freq
            tick_secs   = 1 if TICK_FREQ == "s" else 60
            walk_std    = 0.1 * (tick_secs / 60) ** 0.5

            base        = HOURLY_BASE_PRICE[hour] * (0.85 if is_weekend else 1.0)
            daily_noise = np.random.normal(0, 3.0)
            walk        = np.cumsum(np.random.normal(0, walk_std, n_ticks))
            mid         = base + daily_noise + walk

            half_spread = np.random.uniform(0.5, 2.0, n_ticks)

            # Sell: mid + half_spread + cumulative positive steps (always > buy)
            sell_steps  = np.cumsum(
                np.random.uniform(0.1, 1.0, (n_ticks, N_CIM_LEVELS)), axis=1
            )
            sell_prices = (mid[:, None] + half_spread[:, None] + sell_steps).round(2)
            sell_qtys   = np.random.uniform(0.5, 25.0, (n_ticks, N_CIM_LEVELS)).round(2)

            # Buy: mid - half_spread - cumulative positive steps (always < sell)
            buy_steps  = np.cumsum(
                np.random.uniform(0.1, 1.0, (n_ticks, N_CIM_LEVELS)), axis=1
            )
            buy_prices = (mid[:, None] - half_spread[:, None] - buy_steps).round(2)
            buy_qtys   = np.random.uniform(0.5, 25.0, (n_ticks, N_CIM_LEVELS)).round(2)

            ts_rep = np.repeat(ticks, N_CIM_LEVELS)
            n_half = n_ticks * N_CIM_LEVELS

            dfs.append(pd.DataFrame({
                "timestamp":      np.concatenate([ts_rep,              ts_rep]),
                "delivery_start": delivery_start,
                "delivery_end":   delivery_end,
                "side":           np.concatenate([np.full(n_half, "sell"),
                                                  np.full(n_half, "buy")]),
                "price_eur_mwh":  np.concatenate([sell_prices.ravel(),
                                                  buy_prices.ravel()]),
                "quantity_mwh":   np.concatenate([sell_qtys.ravel(),
                                                  buy_qtys.ravel()]),
            }))

    result = pd.concat(dfs, ignore_index=True)
    result.insert(0, "order_id", result.index + 1)
    return result[["timestamp", "delivery_start", "delivery_end",
                   "side", "price_eur_mwh", "quantity_mwh", "order_id"]]


# ── Validation (mirrors data_loader._validate_spread) ────────────────────────

def _check_spread(df: pd.DataFrame, group_cols: list[str], label: str) -> None:
    sell = (
        df[df["side"] == "sell"]
        .groupby(group_cols)["price_eur_mwh"].min()
        .rename("min_sell")
    )
    buy = (
        df[df["side"] == "buy"]
        .groupby(group_cols)["price_eur_mwh"].max()
        .rename("max_buy")
    )
    merged = pd.concat([sell, buy], axis=1).dropna()
    violations = merged[merged["min_sell"] <= merged["max_buy"]]
    if not violations.empty:
        raise ValueError(
            f"[{label}] Spread violation in {len(violations)} group(s):\n"
            f"{violations.head(5)}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    delivery_days = pd.date_range(
        START_DATE, periods=N_DAYS, freq="D", tz=TIMEZONE
    )

    # Auction curves
    t0 = time.time()
    auction_df = generate_auction_curves(delivery_days)
    auction_df.to_csv(AUCTION_FILE, index=False)
    _check_spread(auction_df, ["delivery_start"], "Auction")

    # CIM order book
    cim_df = generate_cim_order_book(delivery_days)
    cim_df.to_csv(CIM_FILE, index=False)
    _check_spread(cim_df, ["timestamp", "delivery_start"], "CIM")

    elapsed = time.time() - t0

    # ── Summary output ────────────────────────────────────────────────────────
    print(f"Output files:")
    print(f"  {CIM_FILE}")
    print(f"  {AUCTION_FILE}")
    print(f"Row counts:")
    print(f"  CIM     : {len(cim_df):>10,}")
    print(f"  Auction : {len(auction_df):>10,}")
    cim_dates = cim_df["delivery_start"]
    print(f"Date range: {cim_dates.min()}  →  {cim_dates.max()}")
    print(f"Order book validation: PASSED  ({elapsed:.1f}s)")
