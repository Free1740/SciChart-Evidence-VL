"""Interpretable difficulty scoring for chart QA samples.

The score is a weighted sum of four normalised, explainable signals,
each in [0, 1]:

* question length (30%): longer questions tend to ask for more
  involved comparisons / reasoning;
* number of evidence regions (30%): more key evidence to track;
* plotting-code length (20%): more lines usually mean a busier chart;
* number of plotted series (20%): more curves/bars to distinguish.

The total is clipped to [0, 1]. The weights are a starting point for
the pipeline's difficulty filter, not a learned model — they are kept
explicit so the filter can be audited and tuned.
"""

from __future__ import annotations

from .schema import ChartQASample


def _count_plot_calls(plotting_code: str) -> int:
    count = 0
    for line in plotting_code.splitlines():
        stripped = line.strip()
        if ".plot(" in stripped or ".bar(" in stripped or ".scatter(" in stripped:
            count += 1
    return count


def score_difficulty(sample: ChartQASample) -> float:
    """Return a difficulty score in [0, 1] for one sample."""
    question_part = min(len(sample.question) / 120.0, 1.0) * 0.3
    evidence_part = min(len(sample.evidence) / 3.0, 1.0) * 0.3
    code_lines = len(sample.plotting_code.splitlines())
    code_part = min(code_lines / 20.0, 1.0) * 0.2
    plot_part = min(_count_plot_calls(sample.plotting_code) / 5.0, 1.0) * 0.2
    return float(min(1.0, question_part + evidence_part + code_part + plot_part))


def filter_hard(
    samples: list[ChartQASample], threshold: float = 0.7
) -> list[ChartQASample]:
    """Keep only samples with difficulty >= ``threshold``.

    The computed score is written back to each sample's ``difficulty``
    field (including samples that are filtered out), so callers can
    inspect or re-filter without recomputing.
    """
    kept: list[ChartQASample] = []
    for sample in samples:
        sample.difficulty = score_difficulty(sample)
        if sample.difficulty >= threshold:
            kept.append(sample)
    return kept
