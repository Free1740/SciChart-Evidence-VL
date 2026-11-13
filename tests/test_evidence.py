"""Unit tests for src/training/evidence.py.

Requires torch; skipped (not failed) on machines without it:
    pytest.importorskip("torch")
Run with: pytest tests/test_evidence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

torch = pytest.importorskip("torch")

from src.training.evidence import (  # noqa: E402
    apply_evidence_token_mask,
    build_evidence_token_mask,
    evidence_attention_mass,
    evidence_persistence_reward,
)


def test_build_mask_marks_evidence_tokens():
    mask = build_evidence_token_mask(5, [0, 2])
    assert mask.tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_build_mask_ignores_out_of_range_indices():
    mask = build_evidence_token_mask(3, [-1, 1, 3, 99])
    assert mask.tolist() == [0.0, 1.0, 0.0]


def test_build_mask_empty_indices_is_all_zero():
    mask = build_evidence_token_mask(4, [])
    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert mask.dtype == torch.float32


def test_attention_mass_1d():
    attention = torch.tensor([0.1, 0.4, 0.1, 0.4])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # Evidence mass = (0.1 + 0.1) / 1.0 = 0.2.
    assert evidence_attention_mass(attention, mask) == pytest.approx(0.2)


def test_attention_mass_2d_averages_over_rows():
    attention = torch.tensor(
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]
    )
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    # Mean over rows = [0.25, 0.25, 0.25, 0.25]; evidence mass = 0.5.
    assert evidence_attention_mass(attention, mask) == pytest.approx(0.5)


def test_attention_mass_zero_attention_is_zero():
    attention = torch.zeros(4)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert evidence_attention_mass(attention, mask) == 0.0


def test_persistence_reward_constant_attention_has_no_forgetting_penalty():
    mask = torch.tensor([1.0, 0.0])
    trajectory = [torch.tensor([0.8, 0.2]) for _ in range(4)]
    assert evidence_persistence_reward(trajectory, mask) == pytest.approx(0.8)


def test_persistence_reward_penalises_evidence_forgetting():
    mask = torch.tensor([1.0, 0.0])
    trajectory = [
        torch.tensor([0.9, 0.1]),
        torch.tensor([0.9, 0.1]),
        torch.tensor([0.1, 0.9]),
        torch.tensor([0.1, 0.9]),
    ]
    # Masses [0.9, 0.9, 0.1, 0.1]: mean 0.5, first-half mean 0.9,
    # second-half mean 0.1, forgetting 0.8, penalty 0.5 * 0.8 = 0.4.
    assert evidence_persistence_reward(trajectory, mask) == pytest.approx(0.1)


def test_persistence_reward_empty_trajectory_is_zero():
    mask = torch.tensor([1.0, 0.0])
    assert evidence_persistence_reward([], mask) == 0.0


def test_persistence_reward_single_step_has_no_forgetting():
    mask = torch.tensor([1.0, 0.0])
    trajectory = [torch.tensor([0.7, 0.3])]
    assert evidence_persistence_reward(trajectory, mask) == pytest.approx(0.7)


def test_apply_mask_with_zero_boost_is_noop():
    logits = torch.tensor([0.0, 1.0, 2.0])
    mask = torch.tensor([1.0, 0.0, 1.0])
    out = apply_evidence_token_mask(logits, mask, boost=0.0)
    assert torch.equal(out, logits)


def test_apply_mask_with_boost_adds_to_evidence_tokens():
    logits = torch.tensor([0.0, 0.0, 0.0])
    mask = torch.tensor([1.0, 0.0, 1.0])
    out = apply_evidence_token_mask(logits, mask, boost=1.0)
    assert out.tolist() == [1.0, 0.0, 1.0]
