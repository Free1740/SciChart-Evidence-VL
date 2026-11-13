"""Data schemas for chart QA and counterfactual pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class EvidenceRegion:
    """A key visual-evidence region in a scientific chart."""

    label: str  # e.g. "accuracy curve of Model-A", "ablation bar for lr=1e-4"
    kind: Literal["curve", "bar", "point", "text", "region"] = "curve"
    # Normalised bounding box in [0, 1]: (x_min, y_min, x_max, y_max).
    bbox: tuple[float, float, float, float] | None = None
    # Identifier of the evidence inside the plotting code, e.g. a line label
    # or a variable name, used by counterfactual code editing.
    code_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChartQASample:
    """One chart QA sample built from traceable plotting code."""

    sample_id: str
    question: str
    answer: str
    # Original plotting code (e.g. matplotlib) that renders the chart.
    plotting_code: str
    image_path: str | None = None
    # Key visual evidence required to answer the question.
    evidence: list[EvidenceRegion] = field(default_factory=list)
    difficulty: float = 0.0  # 0 (easy) .. 1 (hard), from difficulty filtering
    teacher_reviewed: bool = False
    answer_verified: bool = False
    source: str = ""  # provenance of the plotting code / paper

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChartQASample":
        evidence = [EvidenceRegion(**e) for e in data.get("evidence", [])]
        return cls(**{**data, "evidence": evidence})


@dataclass
class CounterfactualPair:
    """One EP / EA counterfactual pair for a single QA sample."""

    pair_id: str
    sample_id: str
    question: str
    answer: str
    # EP: key evidence preserved, only answer-irrelevant perturbations.
    ep_plotting_code: str
    # EA: key evidence removed (a curve / metric / local data point).
    ea_plotting_code: str
    ep_image_path: str | None = None
    ea_image_path: str | None = None
    # What was changed, e.g. "removed curve 'Model-A'".
    ablation_description: str = ""
    evidence: list[EvidenceRegion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CounterfactualPair":
        evidence = [EvidenceRegion(**e) for e in data.get("evidence", [])]
        return cls(**{**data, "evidence": evidence})
