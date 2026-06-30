"""
Data loading and validation for CIM order book and intraday auction curves.

CIM schema    : timestamp, delivery_start, delivery_end, side, price_eur_mwh,
                quantity_mwh, order_id
Auction schema: auction_time, delivery_start, side, price_eur_mwh,
                quantity_mwh, level
"""

from pathlib import Path

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
