"""Unit tests for src/training/rewards.py.

No torch dependency. Run with: pytest tests/test_rewards.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.rewards import (  # noqa: E402
    combined_reward,
    correctness_reward,
    evidence_consistency_reward,
    format_reward,
)


def test_correctness_exact_match_is_one():
    assert correctness_reward("Model-A", "Model-A") == 1.0


def test_correctness_is_case_and_whitespace_insensitive():
    assert correctness_reward("  model-a  ", "Model-A") == 1.0


def test_correctness_contained_answer_is_half():
    # Gold answer appears inside a longer prediction.
    assert correctness_reward("The answer is Model-A.", "Model-A") == 0.5


def test_correctness_mismatch_is_zero():
    assert correctness_reward("Model-B", "Model-A") == 0.0


def test_correctness_empty_prediction_is_zero():
    assert correctness_reward("", "Model-A") == 0.0
    assert correctness_reward("   ", "Model-A") == 0.0


def test_evidence_consistency_full_when_ep_same_and_ea_differs():
    assert (
        evidence_consistency_reward("Model-A", "Model-A", "Model-B") == 1.0
    )


def test_evidence_consistency_zero_when_ep_differs_and_ea_same():
    assert (
        evidence_consistency_reward("Model-A", "Model-B", "Model-A") == 0.0
    )


def test_evidence_consistency_half_when_only_ep_matches():
    assert (
        evidence_consistency_reward("Model-A", "Model-A", "Model-A") == 0.5
    )


def test_evidence_consistency_half_when_only_ea_differs():
    assert (
        evidence_consistency_reward("Model-A", "Model-B", "Model-B") == 0.5
    )


def test_format_empty_prediction_is_zero():
    assert format_reward("") == 0.0
    assert format_reward("   ") == 0.0


def test_format_normal_prediction_is_one():
    assert format_reward("Model-A has the higher accuracy.") == 1.0


def test_format_runaway_repetition_is_zero():
    assert format_reward("word " * 501) == 0.0


def test_combined_without_counterfactuals():
    # 1.0 (correctness) + 0.1 * 1.0 (format); no evidence term.
    assert combined_reward("Model-A", "Model-A") == 1.1


def test_combined_with_full_evidence_consistency():
    # 1.0 correctness + 0.1 format + 1.0 evidence.
    assert (
        combined_reward("Model-A", "Model-A", prediction_ep="Model-A", prediction_ea="Model-B")
        == 2.1
    )


def test_combined_adds_persistence_reward():
    # 1.0 correctness + 0.1 format + 0.5 persistence.
    assert combined_reward("Model-A", "Model-A", persistence_reward=0.5) == 1.6


def test_combined_ignores_evidence_when_only_one_view_given():
    # Evidence term requires both EP and EA predictions.
    assert (
        combined_reward("Model-A", "Model-A", prediction_ep="Model-A") == 1.1
    )
    assert (
        combined_reward("Model-A", "Model-A", prediction_ea="Model-B") == 1.1
    )


def test_combined_respects_custom_weights():
    result = combined_reward(
        "Model-A",
        "Model-A",
        prediction_ep="Model-A",
        prediction_ea="Model-B",
        weights={"correctness": 0.0, "evidence": 2.0, "format": 0.0},
    )
    # 0.0 correctness + 0.0 format + 2.0 * 1.0 evidence.
    assert result == 2.0
