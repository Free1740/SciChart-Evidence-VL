"""Rule-based Teacher review of chart QA samples.

This is the deterministic, rule-based implementation of the Teacher
review step described in the data pipeline (``scripts/build_dataset.py``):

* a sample is rejected when it is missing the information needed to
  answer it (empty question/answer, no evidence, or evidence that
  cannot be located in the plotting code);
* it is marked ``needs_revision`` when the answer cannot be traced to
  the plotting code or the question (a human / model reviewer should
  look at it before it enters the training set);
* otherwise it is approved.

The interface is deliberately model-free so the pipeline can run
without any external API. A model-based reviewer can replace
:func:`review_sample` later as long as it returns a
:class:`TeacherReview`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .schema import ChartQASample

Verdict = Literal["approve", "reject", "needs_revision"]


@dataclass
class TeacherReview:
    """Result of reviewing one chart QA sample."""

    reviewer: str = "teacher"
    verdict: Verdict = "approve"
    comments: list[str] = field(default_factory=list)
    # 0.0 (unusable) .. 1.0 (clean); 1.0 for approved samples.
    score: float = 1.0


def _evidence_refs(sample: ChartQASample) -> list[str]:
    refs = []
    for region in sample.evidence:
        refs.append(region.code_ref or region.label)
    return refs


def review_sample(sample: ChartQASample, reviewer: str = "teacher") -> TeacherReview:
    """Review one sample with deterministic rules (no external model)."""
    comments: list[str] = []

    # --- Critical checks: any failure means the sample is unusable. ---
    if not sample.question.strip():
        comments.append("Question is empty.")
        return TeacherReview(reviewer, "reject", comments, 0.0)
    if not sample.answer.strip():
        comments.append("Answer is empty.")
        return TeacherReview(reviewer, "reject", comments, 0.0)
    if not sample.plotting_code.strip():
        comments.append("Plotting code is empty.")
        return TeacherReview(reviewer, "reject", comments, 0.0)
    if not sample.evidence:
        comments.append("No key visual-evidence region is annotated.")
        return TeacherReview(reviewer, "reject", comments, 0.0)

    missing = [
        ref
        for ref in _evidence_refs(sample)
        if ref and ref not in sample.plotting_code
    ]
    if missing:
        comments.append(
            "Evidence not found in plotting code: " + ", ".join(missing) + "."
        )
        return TeacherReview(reviewer, "reject", comments, 0.0)

    # --- Soft check: the answer should be traceable to the chart. ---
    answer = sample.answer.strip()
    traceable = answer in sample.plotting_code or answer in sample.question
    if not traceable:
        comments.append(
            "Answer is not literally present in the plotting code or "
            "question; needs a human/model check before use."
        )
        return TeacherReview(reviewer, "needs_revision", comments, 0.7)

    comments.append("Question, answer and evidence are all traceable.")
    return TeacherReview(reviewer, "approve", comments, 1.0)


def review_samples(
    samples: list[ChartQASample], reviewer: str = "teacher"
) -> list[TeacherReview]:
    """Review a list of samples, preserving input order."""
    return [review_sample(sample, reviewer) for sample in samples]
