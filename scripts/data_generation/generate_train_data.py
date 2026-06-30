"""
Generate synthetic training data for the B&P REINFORCE replication.

Market structure:
  - Intraday auction   : clears once per delivery day at 15:00 on D-1
  - Continuous intraday market (CIM): opens at 15:00 D-1, closes 30 min
    before each delivery hour; 5 sell + 5 buy orders per (tick, product)

NOTE: The paper (§VI-A) uses 1-second EPEX SPOT data (TICK_FREQ = "s").
      Generating 200 training days at second resolution requires ~200 GB of
      storage and is impractical for synthetic data.  The default here is
      TICK_FREQ = "min" (1-minute ticks, 30 days).  Set TICK_FREQ = "s" and
      reduce N_DAYS to generate a smaller second-level dataset.

Output: data/train/cim_order_book.csv  and  data/train/intraday_auction_curves.csv

Run from project root:
    venv/bin/python3 scripts/data_generation/generate_train_data.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────────────
SEED       = 42
TIMEZONE   = "Europe/Berlin"
N_DAYS     = 30
START_DATE = "2023-01-01"

# Tick frequency: "s" = 1-second (paper §VI-A), "min" = 1-minute (practical default)
TICK_FREQ = "min"

N_CIM_LEVELS     = 5
N_AUCTION_LEVELS = 10

_PROJECT_ROOT = Path(__file__).parent.parent.parent
CIM_FILE      = _PROJECT_ROOT / "data" / "train" / "cim_order_book.csv"
AUCTION_FILE  = _PROJECT_ROOT / "data" / "train" / "intraday_auction_curves.csv"

# Hourly base price shape (EUR/MWh) – representative German intraday pattern.
# Indexed by delivery hour 0–23.
HOURLY_BASE_PRICE = np.array([
    35, 33, 32, 31, 32, 35,   # 00-06: overnight low
    42, 52, 58, 56, 54, 53,   # 06-12: morning ramp
    52, 51, 52, 54, 58, 62,   # 12-18: afternoon plateau
    65, 63, 58, 52, 45, 40,   # 18-24: evening decline
], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# INTRADAY AUCTION CURVES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_auction_curves(delivery_days: pd.DatetimeIndex) -> pd.DataFrame:
    """
    One auction per delivery day, submitted/cleared at 15:00 D-1.
    Each delivery hour receives N_AUCTION_LEVELS buy and sell price levels.
    Buy prices are always lower than sell prices (uncleared supply/demand curves).
    """
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

            # Mid-price: base pattern + weekend discount + day-to-day noise
            base = HOURLY_BASE_PRICE[hour] * (0.85 if is_weekend else 1.0)
            mid = base + np.random.normal(0, 3.0)

            # Symmetric spread around mid so buy < mid < sell always holds
            half_spread = np.random.uniform(1.0, 3.0)

            # Sell levels: mid + half_spread, then ascending steps
            sell_steps = np.cumsum(np.random.uniform(0.5, 2.0, N_AUCTION_LEVELS))
            sell_prices = (mid + half_spread + sell_steps).round(2)
            sell_qtys = np.random.uniform(1.0, 50.0, N_AUCTION_LEVELS).round(2)

            # Buy levels: mid - half_spread, then descending steps
            buy_steps = np.cumsum(np.random.uniform(0.5, 2.0, N_AUCTION_LEVELS))
            buy_prices = (mid - half_spread - buy_steps).round(2)
            buy_qtys = np.random.uniform(1.0, 50.0, N_AUCTION_LEVELS).round(2)

            for lvl in range(N_AUCTION_LEVELS):
                rows.append((auction_time, delivery_start,
                             "sell", sell_prices[lvl], sell_qtys[lvl], lvl + 1))
                rows.append((auction_time, delivery_start,
                             "buy",  buy_prices[lvl],  buy_qtys[lvl],  lvl + 1))

    return pd.DataFrame(rows, columns=[
        "auction_time", "delivery_start", "side",
        "price_eur_mwh", "quantity_mwh", "level",
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS INTRADAY MARKET (CIM) ORDER BOOK
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cim_order_book(delivery_days: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Tick-level unmatched order book for hourly CIM products (tick = TICK_FREQ).

    Trading window for delivery hour h on day D:
      open  : D-1 15:00
      close : D h:00 − 30 min   (exclusive)

    Unmatched condition guaranteed by construction:
      min(sell) = mid + half_spread + sell_steps[0] > mid + half_spread
                > mid − half_spread > mid − half_spread − buy_steps[0] = max(buy)
    """
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
            delivery_end   = delivery_start + pd.Timedelta(hours=1)
            trading_close  = delivery_start - pd.Timedelta(minutes=30)

            # Tick timestamps in [trading_open, trading_close)
            ticks = pd.date_range(
                start=trading_open,
                end=trading_close,
                freq=TICK_FREQ,
                inclusive="left",
            )
            n_ticks = len(ticks)
            if n_ticks == 0:
                continue

            # Brownian-motion walk scaled so per-minute variance = 0.01 regardless
            # of tick frequency (σ_per_tick = 0.1 × √(tick_seconds / 60)).
            tick_secs = 1 if TICK_FREQ == "s" else 60
            walk_std  = 0.1 * (tick_secs / 60) ** 0.5

            # Mid-price: base + daily noise + intra-session random walk
            base = HOURLY_BASE_PRICE[hour] * (0.85 if is_weekend else 1.0)
            daily_noise = np.random.normal(0, 3.0)
            walk = np.cumsum(np.random.normal(0, walk_std, n_ticks))
            mid = base + daily_noise + walk          # shape (n_ticks,)

            # Per-tick half-spread
            half_spread = np.random.uniform(0.5, 2.0, n_ticks)   # (n_ticks,)

            # Sell side: N_CIM_LEVELS levels strictly above (mid + half_spread)
            sell_steps = np.cumsum(
                np.random.uniform(0.1, 1.0, (n_ticks, N_CIM_LEVELS)), axis=1
            )  # (n_ticks, 5), all positive → sell always > buy
            sell_prices = (mid[:, None] + half_spread[:, None] + sell_steps).round(2)
            sell_qtys   = np.random.uniform(0.5, 25.0, (n_ticks, N_CIM_LEVELS)).round(2)

            # Buy side: N_CIM_LEVELS levels strictly below (mid − half_spread)
            buy_steps = np.cumsum(
                np.random.uniform(0.1, 1.0, (n_ticks, N_CIM_LEVELS)), axis=1
            )  # (n_ticks, 5)
            buy_prices = (mid[:, None] - half_spread[:, None] - buy_steps).round(2)
            buy_qtys   = np.random.uniform(0.5, 25.0, (n_ticks, N_CIM_LEVELS)).round(2)

            # Each tick repeats once per price level (5 sell + 5 buy orders)
            ts_rep = np.repeat(ticks, N_CIM_LEVELS)
            n_half = n_ticks * N_CIM_LEVELS

            df = pd.DataFrame({
                "timestamp":      np.concatenate([ts_rep,              ts_rep]),
                "delivery_start": delivery_start,
                "delivery_end":   delivery_end,
                "side":           np.concatenate([np.full(n_half, "sell"),
                                                  np.full(n_half, "buy")]),
                "price_eur_mwh":  np.concatenate([sell_prices.ravel(),
                                                  buy_prices.ravel()]),
                "quantity_mwh":   np.concatenate([sell_qtys.ravel(),
                                                  buy_qtys.ravel()]),
            })
            dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    # Sequential 1-based order IDs assigned after full concatenation
    result.insert(0, "order_id", result.index + 1)
    return result[["timestamp", "delivery_start", "delivery_end",
                   "side", "price_eur_mwh", "quantity_mwh", "order_id"]]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

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


if __name__ == "__main__":
    np.random.seed(SEED)
    CIM_FILE.parent.mkdir(parents=True, exist_ok=True)

    delivery_days = pd.date_range(
        START_DATE, periods=N_DAYS, freq="D", tz=TIMEZONE
    )

    t0 = time.time()
    auction_df = generate_auction_curves(delivery_days)
    auction_df.to_csv(AUCTION_FILE, index=False)
    _check_spread(auction_df, ["delivery_start"], "Auction")

    cim_df = generate_cim_order_book(delivery_days)
    cim_df.to_csv(CIM_FILE, index=False)
    _check_spread(cim_df, ["timestamp", "delivery_start"], "CIM")

    elapsed = time.time() - t0

    print(f"Output files:")
    print(f"  {CIM_FILE}")
    print(f"  {AUCTION_FILE}")
    print(f"Row counts:")
    print(f"  CIM     : {len(cim_df):>10,}")
    print(f"  Auction : {len(auction_df):>10,}")
    print(f"Date range: {cim_df['delivery_start'].min()}  →  {cim_df['delivery_start'].max()}")
    print(f"Order book validation: PASSED  ({elapsed:.1f}s)")
