"""
Test-set evaluation plots for the REINFORCE threshold policy.

Six figures:
  1. Performance distribution  — histogram + box plot of policy vs RI-LP rewards.
  2. Daily reward time series  — per-day rewards for policy and RI-LP.
  3. Intraday trading behaviour— price landscape + position buildup for 3 example days.
  4. SoC trajectory            — heatmap of end-of-episode SoC + terminal SoC distribution.
  5. Threshold vs market price — scatter of µ_Y vs best_bid and µ_X vs best_ask.
  6. Revenue by delivery hour  — mean net revenue per hour 0-23.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

_POLICY_COL = "#3a7abf"
_RI_COL     = "#e07030"
_SELL_COL   = "#c93030"
_BUY_COL    = "#3060c9"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TickRecord:
    elapsed_hours  : float
    n_active       : int
    mean_bid       : float
    mean_ask       : float
    mean_mu_Y      : float      # mean financial sell threshold across active hours
    mean_mu_X      : float      # mean financial buy threshold across active hours
    total_net_sell : float      # sum(max(0, pos[h])) over all h
    total_net_buy  : float      # sum(max(0, -pos[h])) over all h


@dataclass
class DayResult:
    date           : object
    idx            : int
    policy_reward  : float
    ri_reward      : float
    final_soc      : np.ndarray          # shape (24,) implied SoC at episode end
    revenue_per_hour: np.ndarray         # shape (24,) net EUR per delivery hour
    tick_records   : Optional[list[TickRecord]] = field(default=None)


@dataclass
class ThresholdData:
    mu_Y : list[float] = field(default_factory=list)
    bid  : list[float] = field(default_factory=list)
    mu_X : list[float] = field(default_factory=list)
    ask  : list[float] = field(default_factory=list)
    hour : list[int]   = field(default_factory=list)


# ── Plot 1: Performance distribution ──────────────────────────────────────────

def plot_performance_distribution(
    results : list[DayResult],
    out_path: Path,
) -> None:
    policy   = np.array([r.policy_reward for r in results])
    ri_raw   = np.array([r.ri_reward     for r in results])
    has_ri   = not np.all(np.isnan(ri_raw))
    ri       = ri_raw[~np.isnan(ri_raw)] if has_ri else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Test-set Performance Distribution: Policy" +
                 (" vs RI-LP" if has_ri else ""), fontsize=13)

    # Histogram
    ax = axes[0]
    lo = float(policy.min()) if not has_ri else min(policy.min(), ri.min())
    hi = float(policy.max()) if not has_ri else max(policy.max(), ri.max())
    bins = np.linspace(lo, hi, 25)
    ax.hist(policy, bins=bins, alpha=0.6, color=_POLICY_COL, label="Policy (greedy)")
    if has_ri:
        ax.hist(ri, bins=bins, alpha=0.6, color=_RI_COL, label="RI-LP benchmark")
        ax.axvline(ri.mean(), color=_RI_COL, lw=2, ls="--",
                   label=f"RI-LP mean: {ri.mean():.0f} EUR")
    ax.axvline(policy.mean(), color=_POLICY_COL, lw=2, ls="--",
               label=f"Policy mean: {policy.mean():.0f} EUR")
    ax.set_xlabel("Total episode reward [EUR]")
    ax.set_ylabel("Days")
    ax.set_title("Reward histogram")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)

    # Box plot
    ax = axes[1]
    data       = [policy, ri] if has_ri else [policy]
    ticklabels = ["Policy (greedy)", "RI-LP benchmark"] if has_ri else ["Policy (greedy)"]
    bp = ax.boxplot(data, tick_labels=ticklabels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="white", lw=2))
    bp["boxes"][0].set_facecolor(_POLICY_COL + "aa")
    if has_ri:
        bp["boxes"][1].set_facecolor(_RI_COL + "aa")
    ax.set_ylabel("Total episode reward [EUR]")
    ax.set_title("Box plot")
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 2: Daily reward time series ──────────────────────────────────────────

def plot_daily_rewards(
    results : list[DayResult],
    out_path: Path,
) -> None:
    policy = np.array([r.policy_reward for r in results])
    ri_raw = np.array([r.ri_reward     for r in results])
    has_ri = not np.all(np.isnan(ri_raw))
    xs     = np.arange(1, len(results) + 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(xs, policy, color=_POLICY_COL, lw=1.0, alpha=0.5, label="Policy")
    if has_ri:
        ax.plot(xs, ri_raw, color=_RI_COL, lw=1.0, alpha=0.5, label="RI-LP")

    w = max(1, len(results) // 5)
    k = np.ones(w) / w
    if len(results) >= w:
        policy_roll = np.convolve(policy, k, mode="valid")
        off  = w // 2
        xs_r = np.arange(off + 1, off + len(policy_roll) + 1)
        ax.plot(xs_r, policy_roll, color=_POLICY_COL, lw=2.0,
                label=f"Policy rolling mean (w={w})")
        if has_ri:
            ri_roll = np.convolve(ri_raw, k, mode="valid")
            ax.plot(xs_r, ri_roll, color=_RI_COL, lw=2.0,
                    label=f"RI-LP rolling mean (w={w})")

    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("Test day index")
    ax.set_ylabel("Total episode reward [EUR]")
    ax.set_title("Per-day Reward: Policy" + (" vs RI-LP" if has_ri else "") + " (Test Set)")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 3: Intraday trading behaviour ────────────────────────────────────────

def plot_example_days(
    results : list[DayResult],
    out_path: Path,
) -> None:
    example = [r for r in results if r.tick_records is not None]
    if not example:
        return

    n_rows = len(example)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows),
                             squeeze=False)
    fig.suptitle("Intraday Trading Behaviour — Example Days", fontsize=13)

    for row, res in enumerate(example):
        ticks = res.tick_records
        t     = np.array([tr.elapsed_hours  for tr in ticks])
        mbid  = np.array([tr.mean_bid       for tr in ticks])
        maks  = np.array([tr.mean_ask       for tr in ticks])
        mmuY  = np.array([tr.mean_mu_Y      for tr in ticks])
        mmuX  = np.array([tr.mean_mu_X      for tr in ticks])
        ns    = np.array([tr.n_active       for tr in ticks])
        tsell = np.array([tr.total_net_sell for tr in ticks])
        tbuy  = np.array([tr.total_net_buy  for tr in ticks])
        active = ns > 0

        label = f"Day {res.idx}  |  policy {res.policy_reward:+.0f} EUR  |  RI {res.ri_reward:+.0f} EUR"

        # Left: prices and thresholds
        ax = axes[row][0]
        ax.plot(t[active], mbid[active],  color="#888",      lw=1.0, alpha=0.7, label="mean best_bid")
        ax.plot(t[active], maks[active],  color="#aaa",      lw=1.0, alpha=0.7, ls="--", label="mean best_ask")
        ax.plot(t[active], mmuY[active],  color=_SELL_COL,   lw=1.5, label="mean µ_Y (sell threshold)")
        ax.plot(t[active], mmuX[active],  color=_BUY_COL,    lw=1.5, label="mean µ_X (buy threshold)")
        ax.set_ylabel("EUR/MWh")
        ax.set_title(f"{label}\nPrices & thresholds")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, lw=0.4, alpha=0.5)

        # Right: position buildup
        ax = axes[row][1]
        ax.fill_between(t, tsell,  0, where=tsell > 0,
                        color=_SELL_COL, alpha=0.45, label="net sell (MWh)")
        ax.fill_between(t, -tbuy, 0, where=tbuy  > 0,
                        color=_BUY_COL,  alpha=0.45, label="net buy (MWh)")
        ax.axhline(0, color="#666", lw=0.8)
        ax.set_ylabel("Cumulative MWh")
        ax.set_title("Net committed position over time")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, lw=0.4, alpha=0.5)

        for ax in axes[row]:
            ax.set_xlabel("Elapsed hours since session start (15:00 D-1)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 4: SoC trajectory ────────────────────────────────────────────────────

def plot_soc_trajectory(
    results : list[DayResult],
    out_path: Path,
) -> None:
    soc_matrix = np.stack([r.final_soc for r in results])  # (n_days, 24)
    terminal   = soc_matrix[:, -1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("End-of-episode State of Charge (Test Set)", fontsize=13)

    # Heatmap
    ax = axes[0]
    im = ax.imshow(soc_matrix, aspect="auto", origin="upper",
                   cmap="RdYlGn", vmin=0)
    plt.colorbar(im, ax=ax, label="Implied SoC [MWh]")
    ax.set_xlabel("Delivery hour (0-23)")
    ax.set_ylabel("Test day index")
    ax.set_title("SoC across delivery hours at episode end")
    ax.set_xticks(range(0, 24, 3))

    # Terminal SoC histogram
    ax = axes[1]
    ax.hist(terminal, bins=15, color=_POLICY_COL, alpha=0.75, edgecolor="white")
    ax.axvline(terminal.mean(), color=_POLICY_COL, lw=2, ls="--",
               label=f"Mean: {terminal.mean():.1f} MWh")
    ax.axvline(0, color="#888", lw=1.0, ls=":")
    ax.set_xlabel("Terminal SoC — SoC[hour 23]  [MWh]")
    ax.set_ylabel("Days")
    ax.set_title("Distribution of terminal SoC")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 5: Threshold vs market price ─────────────────────────────────────────

def plot_thresholds_vs_prices(
    td      : ThresholdData,
    out_path: Path,
    max_pts : int = 8000,
) -> None:
    mu_Y = np.array(td.mu_Y)
    bid  = np.array(td.bid)
    mu_X = np.array(td.mu_X)
    ask  = np.array(td.ask)
    hour = np.array(td.hour)

    # Subsample if too many points
    if len(mu_Y) > max_pts:
        idx = np.random.choice(len(mu_Y), max_pts, replace=False)
        mu_Y, bid, mu_X, ask, hour = (
            mu_Y[idx], bid[idx], mu_X[idx], ask[idx], hour[idx]
        )

    cmap   = plt.colormaps["tab20"].resampled(24)
    colors = cmap(hour / 24)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Policy Thresholds vs Market Prices (Test Set)", fontsize=13)

    # Sell side
    ax = axes[0]
    sc = ax.scatter(bid, mu_Y, c=hour, cmap="tab20", vmin=0, vmax=23,
                    s=6, alpha=0.4)
    lo = min(bid.min(), mu_Y.min())
    hi = max(bid.max(), mu_Y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="µ_Y = best_bid")
    ax.set_xlabel("Market best_bid  [EUR/MWh]")
    ax.set_ylabel("µ_Y  (sell threshold)  [EUR/MWh]")
    ax.set_title("Sell side: µ_Y vs best_bid\n(above diagonal → policy willing to sell)")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Delivery hour")

    # Buy side
    ax = axes[1]
    sc = ax.scatter(ask, mu_X, c=hour, cmap="tab20", vmin=0, vmax=23,
                    s=6, alpha=0.4)
    lo = min(ask.min(), mu_X.min())
    hi = max(ask.max(), mu_X.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="µ_X = best_ask")
    ax.set_xlabel("Market best_ask  [EUR/MWh]")
    ax.set_ylabel("µ_X  (buy threshold)  [EUR/MWh]")
    ax.set_title("Buy side: µ_X vs best_ask\n(below diagonal → policy willing to buy)")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Delivery hour")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plot 6: Revenue breakdown by delivery hour ────────────────────────────────

def plot_revenue_by_hour(
    results : list[DayResult],
    out_path: Path,
) -> None:
    rev_matrix = np.stack([r.revenue_per_hour for r in results])  # (n_days, 24)
    mean_rev   = rev_matrix.mean(axis=0)
    std_rev    = rev_matrix.std(axis=0)
    hours      = np.arange(24)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle("Revenue Breakdown by Delivery Hour (Test Set)", fontsize=13)

    # Mean revenue per hour
    ax = axes[0]
    colors = [_SELL_COL if v >= 0 else _BUY_COL for v in mean_rev]
    ax.bar(hours, mean_rev, color=colors, alpha=0.75, width=0.7)
    ax.errorbar(hours, mean_rev, yerr=std_rev,
                fmt="none", color="#333", capsize=3, lw=1.0)
    ax.axhline(0, color="#666", lw=0.8)
    ax.set_xlabel("Delivery hour")
    ax.set_ylabel("Mean net revenue  [EUR]")
    ax.set_title("Mean revenue per delivery hour  (red = net sell, blue = net buy)")
    ax.set_xticks(hours)
    ax.grid(True, axis="y", lw=0.4, alpha=0.5)

    # Revenue heatmap
    ax = axes[1]
    vmax = np.abs(rev_matrix).max() or 1.0
    im = ax.imshow(rev_matrix, aspect="auto", origin="upper",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Net revenue [EUR]")
    ax.set_xlabel("Delivery hour")
    ax.set_ylabel("Test day index")
    ax.set_title("Revenue per hour per day  (red = sell, blue = buy)")
    ax.set_xticks(range(0, 24, 3))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Master entry point ────────────────────────────────────────────────────────

def save_all_plots(
    results  : list[DayResult],
    td       : ThresholdData,
    out_dir  : str | Path,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_performance_distribution(results, out / "fig_test_1_performance_dist.png")
    plot_daily_rewards           (results, out / "fig_test_2_daily_rewards.png")
    plot_example_days            (results, out / "fig_test_3_example_days.png")
    plot_soc_trajectory          (results, out / "fig_test_4_soc_trajectory.png")
    plot_thresholds_vs_prices    (td,      out / "fig_test_5_threshold_vs_price.png")
    plot_revenue_by_hour         (results, out / "fig_test_6_revenue_by_hour.png")

    print(f"[eval_plots] 6 test plots saved to {out}/")
