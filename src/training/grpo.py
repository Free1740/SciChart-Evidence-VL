"""Two-stage GRPO for visual-evidence-driven chart understanding.

Stage 1: KL consistency (original vs EP) + counterfactual separation
         (original vs EA) — suppresses language-prior / surface-cue reliance.
Stage 2: + Evidence Persistence Reward — reduces evidence forgetting in
         long-chain reasoning.

GRPO (Group Relative Policy Optimisation): for each prompt, sample a group
of completions, compute rewards, and use the group-relative advantage
(reward minus group mean, divided by group std) as the policy-gradient
signal, with a KL penalty against the reference model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GRPOConfig:
    group_size: int = 8
    beta_kl: float = 0.04
    clip_epsilon: float = 0.2
    kl_consistency_weight: float = 1.0
    counterfactual_separation_weight: float = 1.0
    evidence_persistence_weight: float = 0.0
    correctness_weight: float = 1.0
    stage: int = 1


def group_relative_advantages(rewards: Tensor) -> Tensor:
    """GRPO advantage: (r - mean) / (std + eps), per group."""
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + 1e-8)


def kl_divergence(logp_policy: Tensor, logp_reference: Tensor) -> Tensor:
    """KL(policy || reference) estimated from log-probs, per token."""
    return torch.exp(logp_reference - logp_policy) - (logp_reference - logp_policy) - 1.0


def kl_consistency_loss(
    logp_original: Tensor, logp_ep: Tensor
) -> Tensor:
    """KL between the prediction distributions on the original and EP views.

    The EP view preserves the key evidence, so the model should predict
    (nearly) the same distribution. Symmetric KL keeps the loss stable.
    """
    p_orig = F.log_softmax(logp_original, dim=-1)
    p_ep = F.log_softmax(logp_ep, dim=-1)
    kl_orig_ep = F.kl_div(p_ep, p_orig, log_target=True, reduction="batchmean")
    kl_ep_orig = F.kl_div(p_orig, p_ep, log_target=True, reduction="batchmean")
    return 0.5 * (kl_orig_ep + kl_ep_orig)


def counterfactual_separation_loss(
    logp_original: Tensor, logp_ea: Tensor, *, margin: float = 1.0
) -> Tensor:
    """Push the EA-view distribution away from the original-view distribution.

    The EA view has the key evidence removed, so a grounded model should
    *not* produce the same prediction. Hinge loss on the KL: the loss is
    zero once the distributions are at least ``margin`` apart.
    """
    p_orig = F.log_softmax(logp_original, dim=-1)
    p_ea = F.log_softmax(logp_ea, dim=-1)
    kl = F.kl_div(p_ea, p_orig, log_target=True, reduction="batchmean")
    return F.relu(margin - kl)


class TwoStageGRPO:
    """Loss computation for both GRPO stages."""

    def __init__(self, config: GRPOConfig) -> None:
        self.config = config

    def stage1_loss(
        self,
        logp_original: Tensor,
        logp_ep: Tensor,
        logp_ea: Tensor,
        logp_reference: Tensor,
        logp_policy: Tensor,
    ) -> dict[str, Tensor]:
        kl_cons = kl_consistency_loss(logp_original, logp_ep)
        separation = counterfactual_separation_loss(logp_original, logp_ea)
        kl_ref = kl_divergence(logp_policy, logp_reference).mean()
        total = (
            self.config.kl_consistency_weight * kl_cons
            + self.config.counterfactual_separation_weight * separation
            + self.config.beta_kl * kl_ref
        )
        return {
            "loss": total,
            "kl_consistency": kl_cons,
            "counterfactual_separation": separation,
            "kl_reference": kl_ref,
        }

    def stage2_loss(
        self,
        stage1_losses: dict[str, Tensor],
        persistence_reward: Tensor,
    ) -> dict[str, Tensor]:
        """Stage 2 adds the evidence-persistence reward (as a loss term)."""
        total = stage1_losses["loss"] - (
            self.config.evidence_persistence_weight * persistence_reward.mean()
        )
        return {**stage1_losses, "loss": total,
                "evidence_persistence_reward": persistence_reward.mean()}

    def total_loss(
        self,
        logp_original: Tensor,
        logp_ep: Tensor,
        logp_ea: Tensor,
        logp_reference: Tensor,
        logp_policy: Tensor,
        persistence_reward: Tensor | None = None,
    ) -> dict[str, Tensor]:
        losses = self.stage1_loss(
            logp_original, logp_ep, logp_ea, logp_reference, logp_policy
        )
        if self.config.stage >= 2 and persistence_reward is not None:
            return self.stage2_loss(losses, persistence_reward)
        return losses
