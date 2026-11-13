"""Reward functions: correctness, evidence consistency, format."""

from __future__ import annotations

import re


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def correctness_reward(prediction: str, gold: str) -> float:
    """1.0 for an exact (normalised) match, 0.5 for gold contained in prediction."""
    pred, answer = _normalise(prediction), _normalise(gold)
    if pred == answer:
        return 1.0
    if answer and answer in pred:
        return 0.5
    return 0.0


def evidence_consistency_reward(
    prediction_original: str, prediction_ep: str, prediction_ea: str
) -> float:
    """Reward predictions that are consistent with EP and change with EA.

    A model grounded in visual evidence should give (nearly) the same
    answer for the original and EP views, but a different answer — or
    lower confidence — for the EA view where the evidence was removed.
    """
    same_ep = 1.0 if _normalise(prediction_original) == _normalise(prediction_ep) else 0.0
    differs_ea = 1.0 if _normalise(prediction_original) != _normalise(prediction_ea) else 0.0
    return 0.5 * same_ep + 0.5 * differs_ea


def format_reward(prediction: str) -> float:
    """Small reward for a well-formed answer (non-empty, no runaway repetition)."""
    if not prediction.strip():
        return 0.0
    words = prediction.split()
    if len(words) > 500:
        return 0.0
    return 1.0


def combined_reward(
    prediction: str,
    gold: str,
    *,
    prediction_ep: str | None = None,
    prediction_ea: str | None = None,
    persistence_reward: float = 0.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted sum of the individual rewards."""
    w = {"correctness": 1.0, "evidence": 1.0, "format": 0.1, "persistence": 1.0}
    if weights:
        w.update(weights)
    total = w["correctness"] * correctness_reward(prediction, gold)
    total += w["format"] * format_reward(prediction)
    total += w["persistence"] * persistence_reward
    if prediction_ep is not None and prediction_ea is not None:
        total += w["evidence"] * evidence_consistency_reward(
            prediction, prediction_ep, prediction_ea
        )
    return float(total)
