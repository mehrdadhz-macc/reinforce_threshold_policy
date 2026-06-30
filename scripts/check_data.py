"""
Quick sanity check for the loaded CIM and auction DataFrames.

Usage:
    venv/bin/python3 scripts/check_data.py [train|test]   (default: train)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_all


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def check_cim(cim):
    section("CIM Order Book")
    print(f"  Rows          : {len(cim):,}")
    print(f"  Columns       : {list(cim.columns)}")
    print(f"  Delivery range: {cim['delivery_start'].min()}  →  {cim['delivery_start'].max()}")
    print(f"  Timestamp range: {cim['timestamp'].min()}  →  {cim['timestamp'].max()}")
    print(f"  Sides         : {cim['side'].value_counts().to_dict()}")
    print(f"  Price range   : [{cim['price_eur_mwh'].min():.2f}, {cim['price_eur_mwh'].max():.2f}] EUR/MWh")
    print(f"  Qty range     : [{cim['quantity_mwh'].min():.2f}, {cim['quantity_mwh'].max():.2f}] MWh")
    print(f"  Nulls         : {cim.isnull().sum().sum()}")

    n_products = cim["delivery_start"].nunique()
    n_timestamps = cim["timestamp"].nunique()
    print(f"  Unique delivery hours : {n_products}")
    print(f"  Unique CIM timestamps : {n_timestamps}")

    # Spread check per (timestamp, delivery_start)
    sell_min = cim[cim["side"] == "sell"].groupby(
        ["timestamp", "delivery_start"])["price_eur_mwh"].min()
    buy_max = cim[cim["side"] == "buy"].groupby(
        ["timestamp", "delivery_start"])["price_eur_mwh"].max()
    spread = (sell_min - buy_max).dropna()
    print(f"  Spread stats (min_sell − max_buy):")
    print(f"    min={spread.min():.4f}  mean={spread.mean():.4f}  max={spread.max():.4f}")
    assert (spread > 0).all(), "FAIL: negative spreads detected in CIM"
    print("  Spread check  : PASS")


def check_auction(auc):
    section("Intraday Auction Curves")
    print(f"  Rows          : {len(auc):,}")
    print(f"  Columns       : {list(auc.columns)}")
    print(f"  Delivery range: {auc['delivery_start'].min()}  →  {auc['delivery_start'].max()}")
    print(f"  Auction times : {auc['auction_time'].nunique()} unique")
    print(f"  Sides         : {auc['side'].value_counts().to_dict()}")
    print(f"  Levels        : {sorted(auc['level'].unique().tolist())}")
    print(f"  Price range   : [{auc['price_eur_mwh'].min():.2f}, {auc['price_eur_mwh'].max():.2f}] EUR/MWh")
    print(f"  Qty range     : [{auc['quantity_mwh'].min():.2f}, {auc['quantity_mwh'].max():.2f}] MWh")
    print(f"  Nulls         : {auc.isnull().sum().sum()}")

    n_products = auc["delivery_start"].nunique()
    print(f"  Unique delivery hours : {n_products}")

    sell_min = auc[auc["side"] == "sell"].groupby("delivery_start")["price_eur_mwh"].min()
    buy_max  = auc[auc["side"] == "buy"].groupby("delivery_start")["price_eur_mwh"].max()
    spread = (sell_min - buy_max).dropna()
    print(f"  Spread stats (min_sell − max_buy):")
    print(f"    min={spread.min():.4f}  mean={spread.mean():.4f}  max={spread.max():.4f}")
    assert (spread > 0).all(), "FAIL: negative spreads detected in Auction"
    print("  Spread check  : PASS")


def check_alignment(cim, auc):
    section("CIM ↔ Auction Alignment")
    cim_products = set(cim["delivery_start"].dt.floor("h").unique())
    auc_products = set(auc["delivery_start"].dt.floor("h").unique())
    common = cim_products & auc_products
    only_cim = cim_products - auc_products
    only_auc = auc_products - cim_products
    print(f"  Delivery hours in both : {len(common)}")
    print(f"  Only in CIM            : {len(only_cim)}")
    print(f"  Only in Auction        : {len(only_auc)}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    print(f"Loading {split} data …")
    cim, auc = load_all(split=split)
    check_cim(cim)
    check_auction(auc)
    check_alignment(cim, auc)
    print("\n  All checks passed.\n")
