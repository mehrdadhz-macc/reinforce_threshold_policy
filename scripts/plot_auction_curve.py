"""
Plot the D-1 intraday auction MID curve for a given day.

Usage:
    venv/bin/python3 scripts/plot_auction_curve.py --file data/train/intraday_auction_curves.csv --day 0
    venv/bin/python3 scripts/plot_auction_curve.py --file data/train/intraday_auction_curves.csv --day 3 --show-regimes

Arguments:
    --file          Path to intraday_auction_curves.csv
    --day           Zero-based day index (default: 0)
    --show-regimes  Overlay buy/sell regime segmentation and p_min/p_max
    --out           Save figure to this path instead of displaying
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_auction
from src.threshold_policy import compute_regimes


def _compute_mids(auc_day: pd.DataFrame, delivery_starts: list) -> np.ndarray:
    sell_best = (
        auc_day[auc_day["side"] == "sell"]
        .groupby("delivery_start")["price_eur_mwh"]
        .min()
    )
    buy_best = (
        auc_day[auc_day["side"] == "buy"]
        .groupby("delivery_start")["price_eur_mwh"]
        .max()
    )
    mids = ((sell_best + buy_best) / 2).to_dict()
    return np.array([mids[ds] for ds in delivery_starts])


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


def plot_auction_curve(
    auc_path   : str | Path,
    day_idx    : int,
    show_regimes: bool = False,
    out_path   : str | Path | None = None,
) -> None:
    auc = load_auction(auc_path)
    auc["delivery_start"] = pd.to_datetime(auc["delivery_start"], utc=True)

    auc["berlin_date"] = auc["delivery_start"].dt.tz_convert("Europe/Berlin").dt.normalize()
    days = sorted(auc["berlin_date"].unique())

    if day_idx < 0 or day_idx >= len(days):
        print(f"Day index {day_idx} out of range (0–{len(days)-1}).")
        sys.exit(1)

    day = days[day_idx]
    auc_day = auc[auc["berlin_date"] == day]
    delivery_starts = sorted(auc_day["delivery_start"].unique().tolist())

    if len(delivery_starts) != 24:
        print(f"Day {day.date()} has {len(delivery_starts)} delivery hours (expected 24). Try another day.")
        sys.exit(1)

    mids = _compute_mids(auc_day, delivery_starts)
    hours = np.arange(24)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, mids, color="steelblue", linewidth=2, marker="o", markersize=4, label="Auction MID")

    if show_regimes:
        regimes = compute_regimes(mids)

        # colour buy-regime segments (split at local maxima)
        buy_bounds = _segment_boundaries(mids, "max")
        sell_bounds = _segment_boundaries(mids, "min")

        cmap_buy  = plt.colormaps["Blues"].resampled(len(buy_bounds))
        cmap_sell = plt.colormaps["Oranges"].resampled(len(sell_bounds))

        for seg_i, (s, e) in enumerate(zip(buy_bounds[:-1], buy_bounds[1:])):
            ax.axvspan(s - 0.5, e + 0.5, alpha=0.12,
                       color=cmap_buy(seg_i + 1), zorder=0)
            # mark p_min of this segment
            seg_min_h = s + int(np.argmin(mids[s:e + 1]))
            ax.axhline(mids[seg_min_h], xmin=(s) / 23, xmax=(e + 1) / 23,
                       color="royalblue", linewidth=1, linestyle="--", alpha=0.6)
            ax.annotate(
                f"p_min={mids[seg_min_h]:.1f}",
                xy=(seg_min_h, mids[seg_min_h]),
                xytext=(0, -14), textcoords="offset points",
                fontsize=7, color="royalblue", ha="center",
            )

        for seg_i, (s, e) in enumerate(zip(sell_bounds[:-1], sell_bounds[1:])):
            # mark p_max of this segment
            seg_max_h = s + int(np.argmax(mids[s:e + 1]))
            ax.annotate(
                f"p_max={mids[seg_max_h]:.1f}",
                xy=(seg_max_h, mids[seg_max_h]),
                xytext=(0, 8), textcoords="offset points",
                fontsize=7, color="darkorange", ha="center",
            )

        # mark local maxima (buy-regime boundaries) and minima (sell-regime boundaries)
        for b in buy_bounds[1:-1]:
            ax.axvline(b, color="royalblue", linewidth=0.8, linestyle=":", alpha=0.7)
        for b in sell_bounds[1:-1]:
            ax.axvline(b, color="darkorange", linewidth=0.8, linestyle=":", alpha=0.7)

        buy_patch  = mpatches.Patch(color="steelblue",  alpha=0.3, label="Buy regime segment")
        pmin_line  = plt.Line2D([0], [0], color="royalblue",  linestyle="--", label="p_min per buy regime")
        pmax_ann   = mpatches.Patch(color="darkorange", alpha=0.0, label="p_max per sell regime (annotated)")
        ax.legend(handles=[ax.lines[0], buy_patch, pmin_line, pmax_ann], fontsize=8)
    else:
        ax.legend(fontsize=9)

    berlin_date = day.tz_convert("Europe/Berlin").date() if hasattr(day, "tz_convert") else str(day.date())
    n_buy  = len(_segment_boundaries(mids, "max")) - 1
    n_sell = len(_segment_boundaries(mids, "min")) - 1
    regime_info = f"  |  {n_buy} buy regimes, {n_sell} sell regimes" if show_regimes else ""
    ax.set_title(f"D-1 Auction MID curve — {berlin_date} (day {day_idx}){regime_info}", fontsize=11)
    ax.set_xlabel("Delivery hour (0 = midnight Berlin)")
    ax.set_ylabel("EUR / MWh")
    ax.set_xticks(hours)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot auction MID curve for a given day")
    parser.add_argument("--file", required=True, help="Path to intraday_auction_curves.csv")
    parser.add_argument("--day",  type=int, default=0, help="Zero-based day index (default: 0)")
    parser.add_argument("--show-regimes", action="store_true",
                        help="Overlay buy/sell regime segmentation")
    parser.add_argument("--out", default=None, help="Save figure to this path instead of displaying")
    args = parser.parse_args()
    plot_auction_curve(args.file, args.day, args.show_regimes, args.out)
