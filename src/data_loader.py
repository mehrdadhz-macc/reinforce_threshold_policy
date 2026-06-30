"""
Data loading and validation for CIM order book and intraday auction curves.

CIM schema    : timestamp, delivery_start, delivery_end, side, price_eur_mwh,
                quantity_mwh, order_id
Auction schema: auction_time, delivery_start, side, price_eur_mwh,
                quantity_mwh, level
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

_TS_COLS_CIM   = ["timestamp", "delivery_start", "delivery_end"]
_TS_COLS_AUC   = ["auction_time", "delivery_start"]
_PROJECT_ROOT  = Path(__file__).parent.parent
_VALID_SPLITS  = {"train", "test"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_timestamps(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _validate_spread(
    df: pd.DataFrame,
    group_cols: list[str],
    context: str,
) -> None:
    """
    For each group defined by group_cols, assert min(sell) > max(buy).
    Raises ValueError listing violating groups.
    """
    sell = (
        df[df["side"] == "sell"]
        .groupby(group_cols)["price_eur_mwh"]
        .min()
        .rename("min_sell")
    )
    buy = (
        df[df["side"] == "buy"]
        .groupby(group_cols)["price_eur_mwh"]
        .max()
        .rename("max_buy")
    )
    merged = pd.concat([sell, buy], axis=1).dropna()
    violations = merged[merged["min_sell"] <= merged["max_buy"]]
    if not violations.empty:
        raise ValueError(
            f"[{context}] Spread violation (min_sell <= max_buy) in "
            f"{len(violations)} group(s):\n{violations.head(10)}"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def load_cim(path: str | Path) -> pd.DataFrame:
    """
    Load and validate the CIM order book.

    Returns a DataFrame with parsed UTC timestamps, sorted by
    (delivery_start, timestamp, side).
    """
    df = pd.read_csv(path)

    required = {"timestamp", "delivery_start", "delivery_end",
                "side", "price_eur_mwh", "quantity_mwh", "order_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CIM file missing columns: {missing}")

    df = _parse_timestamps(df, _TS_COLS_CIM)

    if df[["price_eur_mwh", "quantity_mwh"]].isnull().any().any():
        raise ValueError("CIM contains null prices or quantities.")
    if (df["quantity_mwh"] <= 0).any():
        raise ValueError("CIM contains non-positive quantities.")

    _validate_spread(
        df,
        group_cols=["timestamp", "delivery_start"],
        context="CIM",
    )

    df = df.sort_values(["delivery_start", "timestamp", "side"]).reset_index(drop=True)
    return df


def load_auction(path: str | Path) -> pd.DataFrame:
    """
    Load and validate the intraday auction curves.

    Returns a DataFrame with parsed UTC timestamps, sorted by
    (delivery_start, side, level).
    """
    df = pd.read_csv(path)

    required = {"auction_time", "delivery_start", "side",
                "price_eur_mwh", "quantity_mwh", "level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Auction file missing columns: {missing}")

    df = _parse_timestamps(df, _TS_COLS_AUC)

    if df[["price_eur_mwh", "quantity_mwh"]].isnull().any().any():
        raise ValueError("Auction contains null prices or quantities.")
    if (df["quantity_mwh"] <= 0).any():
        raise ValueError("Auction contains non-positive quantities.")

    _validate_spread(
        df,
        group_cols=["delivery_start"],
        context="Auction",
    )

    df = df.sort_values(["delivery_start", "side", "level"]).reset_index(drop=True)
    return df


def build_day_index(
    cim: pd.DataFrame,
    auc: pd.DataFrame,
) -> list[tuple]:
    """
    Return a sorted list of (berlin_day, delivery_starts, session_start_utc).

    Only days with exactly 24 delivery hours present in both CIM and auction
    data are included.  session_start = 15:00 D-1 Berlin time expressed in UTC.
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
        session_berlin = (day - pd.Timedelta(days=1)).replace(
            hour=15, minute=0, second=0
        )
        session_utc = session_berlin.tz_convert("UTC")
        days.append((day, delivery_starts, session_utc))

    return sorted(days, key=lambda x: x[0])


def day_auction_mids(
    auc            : pd.DataFrame,
    delivery_starts: list,
) -> np.ndarray | None:
    """
    Compute per-hour D-1 auction MID = (best_ask + best_bid) / 2.

    Returns shape-(24,) array, or None if any delivery hour is missing.
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


def load_all(
    split: str = "train",
    *,
    cim_path: str | Path | None = None,
    auction_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate both datasets.

    Args:
        split        : "train" or "test" — selects data/<split>/ automatically.
        cim_path     : explicit path override; if given, `split` is ignored.
        auction_path : explicit path override; if given, `split` is ignored.

    Returns:
        (cim_df, auction_df)
    """
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {_VALID_SPLITS}; got {split!r}")
    data_dir = _PROJECT_ROOT / "data" / split
    if cim_path is None:
        cim_path = data_dir / "cim_order_book.csv"
    if auction_path is None:
        auction_path = data_dir / "intraday_auction_curves.csv"
    return load_cim(cim_path), load_auction(auction_path)
