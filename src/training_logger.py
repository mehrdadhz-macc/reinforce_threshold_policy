"""
Training logger and plotter for the REINFORCE threshold policy.

Records per-episode metrics and per-rep (full-data-pass) parameter snapshots,
then produces four figures after training:

  Figure 1 — Policy parameters (alpha pairs by step, one panel per pair).
  Figure 2 — Sampling distribution widths (sigma_X, sigma_Y).
  Figure 3 — Per-episode total reward + rolling mean; vertical bands = rep boundaries.
  Figure 4 — Per-episode REINFORCE loss.

The x-axis for Figures 1 & 2 is "rep index" (one tick per full pass through
the training data).  Figures 3 & 4 use episode index on the x-axis with
rep boundaries drawn as vertical lines and rep-mean reward overlaid.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Colour palette ─────────────────────────────────────────────────────────────
_SELL_COL = "#e05c5c"   # red-ish for sell / X parameters
_BUY_COL  = "#5c8ae0"   # blue-ish for buy  / Y parameters
_PHASE_COLOURS = ["#d4e8d4", "#d4d4e8", "#e8d4d4"]   # hourly / 15-min / 5-min


class TrainingLogger:
    """Collects training metrics and renders summary figures."""

    def __init__(self) -> None:
        # Per-episode records
        self._ep_rewards : list[float] = []
        self._ep_losses  : list[float] = []
        self._ep_phases  : list[str]   = []   # phase label for each episode
        self._ep_reps    : list[int]   = []   # global rep index for each episode

        # Per-rep records (one entry per complete pass through training data)
        self._rep_mean_rewards : list[float]            = []
        self._rep_params       : list[dict[str, float]] = []
        self._rep_phases       : list[str]              = []
        self._rep_labels       : list[str]              = []   # e.g. "hourly rep 1"

        self._global_rep = 0   # incremented by log_rep()

    # ── Ingestion API ──────────────────────────────────────────────────────────

    def log_episode(
        self,
        phase  : str,
        reward : float,
        loss   : float,
    ) -> None:
        """Call after each episode completes (inside _run_phase)."""
        self._ep_rewards.append(float(reward))
        self._ep_losses.append(float(loss))
        self._ep_phases.append(phase)
        self._ep_reps.append(self._global_rep)

    def log_rep(
        self,
        phase         : str,
        local_rep     : int,
        mean_reward   : float,
        param_snapshot: dict[str, float],
    ) -> None:
        """Call once after every complete pass through the training days."""
        self._global_rep += 1
        self._rep_mean_rewards.append(float(mean_reward))
        self._rep_params.append(dict(param_snapshot))
        self._rep_phases.append(phase)
        self._rep_labels.append(f"{phase} rep {local_rep}")

    # ── Plotting API ───────────────────────────────────────────────────────────

    def plot(self, out_dir: str | Path = "outputs/plots") -> None:
        """Generate and save all four figures to *out_dir*."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if not self._rep_params:
            print("[TrainingLogger] No data recorded — skipping plots.")
            return

        self._plot_alpha_params(out / "fig1_policy_params.png")
        self._plot_sigma_params(out / "fig2_sigma_params.png")
        self._plot_rewards(out / "fig3_rewards.png")
        self._plot_loss(out / "fig4_loss.png")
        print(f"[TrainingLogger] Plots saved to {out}/")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _phase_x_ranges(self) -> list[tuple[int, int, str]]:
        """Return [(start_rep, end_rep, phase_label)] for background shading."""
        ranges: list[tuple[int, int, str]] = []
        if not self._rep_phases:
            return ranges
        cur_phase = self._rep_phases[0]
        start = 0
        for i, ph in enumerate(self._rep_phases):
            if ph != cur_phase:
                ranges.append((start, i, cur_phase))
                cur_phase = ph
                start = i
        ranges.append((start, len(self._rep_phases), cur_phase))
        return ranges

    def _shade_phases(self, ax: plt.Axes, n_reps: int) -> None:
        unique_phases = list(dict.fromkeys(self._rep_phases))
        col_map = {ph: _PHASE_COLOURS[i % len(_PHASE_COLOURS)]
                   for i, ph in enumerate(unique_phases)}
        for start, end, ph in self._phase_x_ranges():
            ax.axvspan(start, end, alpha=0.25, color=col_map[ph], zorder=0)

    def _shade_phases_ep(self, ax: plt.Axes) -> None:
        """Shade episode-axis plots by phase using rep boundary episode indices."""
        if not self._ep_reps:
            return
        unique_phases = list(dict.fromkeys(self._rep_phases))
        col_map = {ph: _PHASE_COLOURS[i % len(_PHASE_COLOURS)]
                   for i, ph in enumerate(unique_phases)}
        # Find episode index where each rep starts
        rep_start_ep: dict[int, int] = {}
        for ep_i, rep in enumerate(self._ep_reps):
            if rep not in rep_start_ep:
                rep_start_ep[rep] = ep_i
        for start_rep, end_rep, ph in self._phase_x_ranges():
            ep_start = rep_start_ep.get(start_rep, 0)
            if end_rep < len(self._rep_phases):
                ep_end = rep_start_ep.get(end_rep, len(self._ep_rewards))
            else:
                ep_end = len(self._ep_rewards)
            ax.axvspan(ep_start, ep_end, alpha=0.2, color=col_map[ph], zorder=0)

    def _plot_alpha_params(self, path: Path) -> None:
        pairs = [
            ("alpha_s1", "alpha_b1", "Step 1 — anchor  α₁"),
            ("alpha_s2", "alpha_b2", "Step 2 — SoC pressure  α₂"),
            ("alpha_s3", "alpha_b3", "Step 3 — urgency magnitude  α₃"),
            ("alpha_s4", "alpha_b4", "Step 3 — urgency rate  α₄"),
            ("alpha_s5", "alpha_b5", "Step 4 — RI auxiliary shift  α₅"),
        ]
        n_reps = len(self._rep_params)
        xs = np.arange(1, n_reps + 1)

        fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
        fig.suptitle("Policy Parameters vs Training Reps", fontsize=13, y=1.01)

        for ax, (s_key, b_key, title) in zip(axes, pairs):
            s_vals = [p[s_key] for p in self._rep_params]
            b_vals = [p[b_key] for p in self._rep_params]
            self._shade_phases(ax, n_reps)
            ax.plot(xs, s_vals, color=_SELL_COL, lw=1.5, marker="o", ms=3,
                    label=f"sell  ({s_key})")
            ax.plot(xs, b_vals, color=_BUY_COL,  lw=1.5, marker="s", ms=3,
                    label=f"buy   ({b_key})")
            ax.set_ylabel("value", fontsize=8)
            ax.set_title(title, fontsize=9, loc="left")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, lw=0.4, alpha=0.5)

        axes[-1].set_xlabel("Rep (one full pass through training days)")
        self._add_phase_legend(axes[0])
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_sigma_params(self, path: Path) -> None:
        n_reps = len(self._rep_params)
        xs     = np.arange(1, n_reps + 1)
        sx = [p["sigma_X"] for p in self._rep_params]
        sy = [p["sigma_Y"] for p in self._rep_params]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle("Sampling Distribution Widths vs Training Reps", fontsize=13)

        for ax, vals, col, label in [
            (axes[0], sx, _SELL_COL, "σ_X  (buy / charge distribution)"),
            (axes[1], sy, _BUY_COL,  "σ_Y  (sell / discharge distribution)"),
        ]:
            self._shade_phases(ax, n_reps)
            ax.plot(xs, vals, color=col, lw=1.5, marker="o", ms=4)
            ax.set_ylabel("σ  [EUR/MWh]", fontsize=9)
            ax.set_title(label, fontsize=9, loc="left")
            ax.grid(True, lw=0.4, alpha=0.5)

        axes[-1].set_xlabel("Rep (one full pass through training days)")
        self._add_phase_legend(axes[0])
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_rewards(self, path: Path) -> None:
        n_ep  = len(self._ep_rewards)
        xs_ep = np.arange(1, n_ep + 1)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False,
                                 gridspec_kw={"height_ratios": [2, 1]})

        # ── Top: per-episode reward + rolling mean ─────────────────────────────
        ax = axes[0]
        self._shade_phases_ep(ax)
        ax.scatter(xs_ep, self._ep_rewards, s=4, alpha=0.4, color="#888",
                   label="episode reward")
        window = max(1, n_ep // 40)
        kernel  = np.ones(window) / window
        rolling = np.convolve(self._ep_rewards, kernel, mode="valid")
        offset  = window // 2
        ax.plot(np.arange(offset + 1, offset + len(rolling) + 1),
                rolling, color="#333", lw=1.5, label=f"rolling mean (w={window})")
        # vertical lines at rep boundaries
        rep_starts: dict[int, int] = {}
        for ep_i, rep in enumerate(self._ep_reps):
            if rep not in rep_starts:
                rep_starts[rep] = ep_i + 1
        for rep, ep_start in sorted(rep_starts.items())[1:]:
            ax.axvline(ep_start, color="#aaa", lw=0.6, ls="--")
        ax.set_ylabel("Total reward  [EUR]", fontsize=9)
        ax.set_title("Per-episode total reward", fontsize=10, loc="left")
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.set_xlabel("Episode index")

        # ── Bottom: per-rep mean reward ────────────────────────────────────────
        ax2 = axes[1]
        n_reps = len(self._rep_mean_rewards)
        xs_rep = np.arange(1, n_reps + 1)
        self._shade_phases(ax2, n_reps)
        ax2.bar(xs_rep, self._rep_mean_rewards,
                color=[_BUY_COL if r >= 0 else _SELL_COL
                       for r in self._rep_mean_rewards],
                alpha=0.7, width=0.6)
        ax2.axhline(0, color="#333", lw=0.8)
        ax2.set_ylabel("Mean reward  [EUR]", fontsize=9)
        ax2.set_title("Per-rep mean reward (one bar = one full pass)", fontsize=9, loc="left")
        ax2.set_xlabel("Rep (one full pass through training days)")
        ax2.grid(True, axis="y", lw=0.4, alpha=0.5)
        self._add_phase_legend(ax2)

        fig.suptitle("Reward Progress During Training", fontsize=13)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_loss(self, path: Path) -> None:
        n_ep  = len(self._ep_losses)
        xs_ep = np.arange(1, n_ep + 1)

        fig, ax = plt.subplots(figsize=(12, 4))
        self._shade_phases_ep(ax)
        ax.plot(xs_ep, self._ep_losses, lw=0.8, alpha=0.6, color="#555",
                label="episode loss")
        window  = max(1, n_ep // 40)
        kernel  = np.ones(window) / window
        rolling = np.convolve(self._ep_losses, kernel, mode="valid")
        offset  = window // 2
        ax.plot(np.arange(offset + 1, offset + len(rolling) + 1),
                rolling, color="#c04040", lw=1.5, label=f"rolling mean (w={window})")
        # rep boundaries
        rep_starts: dict[int, int] = {}
        for ep_i, rep in enumerate(self._ep_reps):
            if rep not in rep_starts:
                rep_starts[rep] = ep_i + 1
        for rep, ep_start in sorted(rep_starts.items())[1:]:
            ax.axvline(ep_start, color="#aaa", lw=0.6, ls="--")
        ax.set_xlabel("Episode index")
        ax.set_ylabel("REINFORCE loss", fontsize=9)
        ax.set_title("Training Loss", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)
        self._add_phase_legend(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _add_phase_legend(self, ax: plt.Axes) -> None:
        unique_phases = list(dict.fromkeys(self._rep_phases))
        patches = [
            mpatches.Patch(facecolor=_PHASE_COLOURS[i % len(_PHASE_COLOURS)],
                           alpha=0.5, label=ph)
            for i, ph in enumerate(unique_phases)
        ]
        if patches:
            ax.legend(handles=patches + ax.get_legend_handles_labels()[0],
                      labels=[p.get_label() for p in patches]
                             + ax.get_legend_handles_labels()[1],
                      fontsize=7, loc="upper left")
