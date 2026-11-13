"""Unit tests for src/data/counterfactual.py.

No torch dependency. Run with: pytest tests/test_counterfactual.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.counterfactual import (  # noqa: E402
    EXAMPLE_CODE,
    CounterfactualBuilder,
    make_evidence_ablated,
    make_evidence_preserved,
)
from src.data.schema import ChartQASample, EvidenceRegion  # noqa: E402


def _sample(**overrides):
    data = {
        "sample_id": "demo-001",
        "question": "Which model has higher accuracy at x=3?",
        "answer": "Model-A",
        "plotting_code": EXAMPLE_CODE,
        "evidence": [EvidenceRegion(label="Model-A", code_ref="Model-A")],
    }
    data.update(overrides)
    return ChartQASample(**data)


def test_ep_preserves_key_curve():
    evidence = EvidenceRegion(label="Model-A", code_ref="Model-A")
    ep = make_evidence_preserved(EXAMPLE_CODE, evidence)
    assert "Model-A" in ep.edited_code
    # The evidence curve itself is untouched.
    assert 'plt.plot(x, [0.8, 0.85, 0.9], label="Model-A", color="blue")' in ep.edited_code


def test_ep_perturbs_only_irrelevant_curve():
    evidence = EvidenceRegion(label="Model-A", code_ref="Model-A")
    ep = make_evidence_preserved(EXAMPLE_CODE, evidence)
    # Non-evidence curve (Model-B) is recoloured to gray.
    assert 'color="gray"' in ep.edited_code
    assert 'label="Model-B", color="red"' not in ep.edited_code


def test_ep_with_nothing_irrelevant_to_perturb_is_unchanged():
    code = 'plt.plot(x, [0.8, 0.85, 0.9], label="Model-A", color="blue")\n'
    evidence = EvidenceRegion(label="Model-A", code_ref="Model-A")
    ep = make_evidence_preserved(code, evidence)
    assert ep.edited_code == code


def test_ea_removes_key_curve():
    evidence = EvidenceRegion(label="Model-A", code_ref="Model-A")
    ea = make_evidence_ablated(EXAMPLE_CODE, evidence)
    assert "Model-A" not in ea.edited_code
    # The other curve survives the ablation.
    assert "Model-B" in ea.edited_code


def test_ea_raises_when_evidence_not_in_code():
    evidence = EvidenceRegion(label="NoSuchCurve", code_ref="NoSuchCurve")
    with pytest.raises(ValueError, match="not found in plotting code"):
        make_evidence_ablated(EXAMPLE_CODE, evidence)


def test_build_pair_ep_keeps_and_ea_removes_evidence():
    pair = CounterfactualBuilder().build_pair(_sample())
    assert pair.pair_id == "demo-001-cf"
    assert pair.sample_id == "demo-001"
    assert pair.answer == "Model-A"
    assert "Model-A" in pair.ep_plotting_code
    assert "Model-A" not in pair.ea_plotting_code
    assert "Model-A" in pair.ablation_description


def test_build_pair_raises_without_evidence():
    with pytest.raises(ValueError, match="has no evidence region"):
        CounterfactualBuilder().build_pair(_sample(evidence=[]))


def test_build_pairs_skips_unlocatable_samples():
    good = _sample()
    bad = _sample(
        sample_id="bad-001",
        evidence=[EvidenceRegion(label="NoSuchCurve", code_ref="NoSuchCurve")],
    )
    pairs = CounterfactualBuilder().build_pairs([good, bad])
    assert len(pairs) == 1
    assert pairs[0].sample_id == "demo-001"


def test_counterfactual_pair_round_trip():
    pair = CounterfactualBuilder().build_pair(_sample())
    restored = type(pair).from_dict(pair.to_dict())
    assert restored.pair_id == pair.pair_id
    assert restored.answer == pair.answer
    assert restored.ep_plotting_code == pair.ep_plotting_code
    assert restored.ea_plotting_code == pair.ea_plotting_code
    assert restored.evidence[0].label == "Model-A"
