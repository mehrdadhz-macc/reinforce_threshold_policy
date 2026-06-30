"""
REINFORCE trainer for multi-hour daily episodes (Bertrand & Papavasiliou 2020, §IV–V).

Episode = one full trading day (24 delivery hours traded jointly).

At each step t the agent selects one signed-quantity action per active hour d.
The joint log-prob at step t = Σ_d log P(a_{t,d} | s_t) over all active hours
(using CDF-difference formula for discrete quantity levels, §IV-A, Appendix C).

REINFORCE return g_t is the total undiscounted future reward from t to end of day.

Update rule (B&P Eq. 2):
    α ← α + ρ · ∇_α log π_α(a|s) · g_t
    Adam used as the step-size schedule.
    Returns are undiscounted and used without any normalisation (§IV).
    Gradient flows through every step including idle ones — infeasible actions
    are handled by probability mass reallocation in forward_batch (§V-F).
"""

from __future__ import annotations

import numpy as np
import torch

from src.environment import StepState
from src.threshold_policy import AlphaPolicy, compute_regimes


class REINFORCEAgent:
    def __init__(
        self,
        lr             : float = 1e-3,
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
        self.policy    = AlphaPolicy(
            alpha_s1_init, alpha_b1_init,
            alpha_s2_init, alpha_b2_init,
            alpha_s3_init, alpha_b3_init,
            alpha_s4_init, alpha_b4_init,
            alpha_s5_init, alpha_b5_init,
            log_sigma_init,
            round_trip_eff=round_trip_eff,
            n_levels=n_levels,
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self._log_probs: list[torch.Tensor] = []
        self._rewards  : list[float]         = []

    # ── Episode setup ─────────────────────────────────────────────────────────

    def set_episode(
        self,
        auction_mids: np.ndarray,
        capacity    : float = 1.0,
        initial_soc : float = 0.0,
        max_charge  : float = 1.0,
        max_dis     : float = 1.0,
    ) -> None:
        """Compute buy/sell regimes from the 24-hour auction MID curve."""

        # first compute the buy and sell regimes 
        regime = compute_regimes(auction_mids)
        self.policy.set_regime_info(regime)
        self.policy.set_env_params(capacity, initial_soc, max_charge, max_dis)

    # ── Per-step interface ────────────────────────────────────────────────────

    def select_actions(
        self, state: StepState
    ) -> tuple[dict[int, float], torch.Tensor]:
        """
        Sample one signed-quantity action per active delivery hour.

        Returns:
            actions      : {delivery_hour → signed_quantity_mwh}
            step_log_prob: joint log-prob tensor for store_step()
        """
        actions, step_log_prob = self.policy.forward_batch(
            state.active_hours,
            state.best_bid,
            state.best_ask,
            state.order_book,
            state.elapsed_hours,
            state.v_end_fraction,
            position=state.position,
            step_hours=state.step_hours,
        )
        return actions, step_log_prob

    def store_step(
        self,
        step_log_prob: torch.Tensor,
        reward       : float,
        info         : dict[int, dict],
    ) -> None:
        """
        Accumulate one step's log-prob and reward.

        Gradient flows through every step including idle ones: infeasible levels
        have their probability mass redirected by _reweight_feasibility in
        forward_batch (§V-F), so P(idle) already carries the correct mass and
        its gradient is meaningful for REINFORCE.
        """
        self._log_probs.append(step_log_prob)
        self._rewards.append(float(reward))

    # ── End-of-episode update ─────────────────────────────────────────────────

    def update(self, grad_clip: float = 1.0) -> float:
        """Apply one REINFORCE gradient step. Returns loss scalar; clears buffers."""
        if len(self._log_probs) != len(self._rewards):
            raise RuntimeError("log_prob / reward buffer length mismatch.")

        returns = self._episode_returns()
        loss    = torch.stack(
            [-lp * G for lp, G in zip(self._log_probs, returns)]
        ).sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
        self.optimizer.step()

        loss_val = loss.item()
        self._log_probs.clear()
        self._rewards.clear()
        return loss_val

    # ── Evaluation ────────────────────────────────────────────────────────────

    def greedy_actions(self, state: StepState, fast: bool = False) -> dict[int, float]:
        """Deterministic actions at threshold means; for evaluation (no gradient).

        fast=True skips the per-tick RI LP (uses auction-mid reference instead),
        giving 10-100× speedup with negligible accuracy loss for evaluation.
        """
        return self.policy.greedy_batch(
            state.active_hours,
            state.best_bid,
            state.best_ask,
            state.order_book,
            state.elapsed_hours,
            state.v_end_fraction,
            position=state.position,
            step_hours=state.step_hours,
            fast=fast,
        )

    def param_snapshot(self) -> dict[str, float]:
        return self.policy.param_snapshot()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _episode_returns(self) -> torch.Tensor:
        """Undiscounted returns g_t = Σ_{k≥t} r_k (B&P Eq. 2, no normalisation)."""
        G = 0.0
        returns: list[float] = []
        for r in reversed(self._rewards):
            G = G + r
            returns.append(G)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)
