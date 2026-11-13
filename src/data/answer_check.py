"""Rule-based answer verification for chart QA samples.

:func:`verify_answer` is the interface used by the data pipeline
(``scripts/build_dataset.py``). The current implementation is a
deterministic rule check, not a model judgement:

* the answer must be non-empty;
* the answer text itself, or the label of the key evidence region,
  must appear in the plotting code or in the question — i.e. the
  answer has to be traceable to something the chart actually shows.

A model-based verifier (e.g. asking a Teacher model whether the
answer follows from the chart) can replace this function later as
long as it keeps the same signature ``verify_answer(sample) -> bool``.
"""

from __future__ import annotations

from .schema import ChartQASample


def _evidence_labels(sample: ChartQASample) -> list[str]:
    labels = []
    for region in sample.evidence:
        labels.append(region.label)
        if region.code_ref:
            labels.append(region.code_ref)
    return labels


def verify_answer(sample: ChartQASample) -> bool:
    """Return True when the answer is traceable to the chart/question."""
    answer = sample.answer.strip()
    if not answer:
        return False
    if not sample.question.strip() or not sample.plotting_code.strip():
        return False
    # Direct trace: the answer text itself is in the code or question.
    if answer in sample.plotting_code or answer in sample.question:
        return True
    # Indirect trace: the key evidence label is in the code/question,
    # so the answer refers to a real visual element of the chart.
    for label in _evidence_labels(sample):
        if label and (label in sample.plotting_code or label in sample.question):
            return True
    return False


def verify_answers(samples: list[ChartQASample]) -> dict[str, bool]:
    """Verify a list of samples, keyed by ``sample_id``."""
    return {sample.sample_id: verify_answer(sample) for sample in samples}
