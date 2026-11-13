"""Evaluation over 8 public benchmarks, with an averaged score."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# 8 benchmarks: 7 chart-understanding + 1 general visual-reasoning.
BENCHMARKS: list[str] = [
    "ChartQA",
    "PlotQA",
    "ChartBench",
    "CharXiv",
    "FigureQA",
    "DVQA",
    "ChartX",
    "MathVista",
]

# Reported averages for reference (baseline Qwen2.5-VL vs this method).
BASELINE_AVERAGE = 44.3
REPORTED_AVERAGE = 52.1


@dataclass
class BenchmarkResult:
    name: str
    score: float  # 0..100
    num_samples: int = 0

    def to_dict(self) -> dict:
        return {"name": self.name, "score": self.score, "num_samples": self.num_samples}


def average_score(results: list[BenchmarkResult]) -> float:
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def evaluate_benchmark(
    name: str,
    predict: Callable[[str, str], str],
    data_path: str | Path,
) -> BenchmarkResult:
    """Evaluate one benchmark from a JSONL file of {image, question, answer}.

    ``predict(image_path, question)`` is supplied by the caller (the model
    wrapper); this function only handles loading, scoring and aggregation.
    """
    from ..training.rewards import correctness_reward  # local import to avoid cycles

    path = Path(data_path)
    if not path.exists():
        return BenchmarkResult(name=name, score=0.0, num_samples=0)
    total, correct = 0, 0.0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pred = predict(item["image"], item["question"])
            correct += correctness_reward(pred, item["answer"])
            total += 1
    score = 100.0 * correct / total if total else 0.0
    return BenchmarkResult(name=name, score=score, num_samples=total)


def run_evaluation(
    benchmarks: list[str],
    predict: Callable[[str, str], str],
    data_dir: str | Path,
    output_path: str | Path,
) -> dict:
    """Run all benchmarks and write a JSON report with the average score."""
    results = [
        evaluate_benchmark(name, predict, Path(data_dir) / f"{name}.jsonl")
        for name in benchmarks
    ]
    report = {
        "benchmarks": [r.to_dict() for r in results],
        "average": average_score(results),
        "baseline_average": BASELINE_AVERAGE,
        "reported_average": REPORTED_AVERAGE,
        "improvement": REPORTED_AVERAGE - BASELINE_AVERAGE,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
