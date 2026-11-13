"""Evidence Token Mask and Evidence Persistence Reward.

Idea: during long-chain reasoning the model should keep attending to the
key visual-evidence tokens. ``build_evidence_token_mask`` marks which
image tokens belong to the key evidence region; the persistence reward
measures how much attention those tokens keep receiving over reasoning
steps, and penalises forgetting (a drop in later steps).
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_evidence_token_mask(
    num_image_tokens: int,
    evidence_token_indices: list[int],
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return a (num_image_tokens,) mask: 1 on evidence tokens, 0 elsewhere."""
    mask = torch.zeros(num_image_tokens, dtype=dtype)
    for idx in evidence_token_indices:
        if 0 <= idx < num_image_tokens:
            mask[idx] = 1.0
    return mask


def evidence_attention_mass(attention: Tensor, evidence_mask: Tensor) -> float:
    """Mean attention mass on evidence tokens for one step.

    ``attention`` is (seq, seq) or (seq,) attention weights;
    ``evidence_mask`` is (num_image_tokens,) with 1 on evidence tokens.
    """
    if attention.dim() == 2:
        # Attention paid by generated tokens to image tokens: average over rows.
        attn_to_image = attention.mean(dim=0)[: evidence_mask.numel()]
    else:
        attn_to_image = attention[: evidence_mask.numel()]
    total = attn_to_image.sum()
    if total <= 0:
        return 0.0
    return float((attn_to_image * evidence_mask).sum() / total)


def evidence_persistence_reward(
    trajectory: list[Tensor],
    evidence_mask: Tensor,
    *,
    forgetting_penalty: float = 0.5,
) -> float:
    """Reward for keeping visual evidence alive across reasoning steps.

    ``trajectory`` is a list of per-step attention tensors (one per
    reasoning step). Reward = mean evidence attention mass, minus a
    penalty for the drop from the first half to the second half of the
    chain (evidence forgetting).
    """
    if not trajectory:
        return 0.0
    masses = [evidence_attention_mass(attn, evidence_mask) for attn in trajectory]
    mean_mass = sum(masses) / len(masses)
    if len(masses) >= 2:
        half = len(masses) // 2
        first = sum(masses[:half]) / half
        second = sum(masses[half:]) / (len(masses) - half)
        forgetting = max(0.0, first - second)
    else:
        forgetting = 0.0
    return float(mean_mass - forgetting_penalty * forgetting)


def apply_evidence_token_mask(
    logits: Tensor, evidence_mask: Tensor, *, boost: float = 0.0
) -> Tensor:
    """Optionally boost logits of evidence tokens during decoding.

    This is a no-op when ``boost == 0``; with ``boost > 0`` it adds a
    small bonus to evidence-token logits to keep the decoder grounded.
    Kept as an explicit, testable helper rather than hidden in training.
    """
    if boost == 0.0:
        return logits
    return logits + boost * evidence_mask.to(logits.device, logits.dtype)
