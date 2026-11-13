"""Build a filtered chart QA dataset: review -> verify -> difficulty filter.

Pipeline (mirrors the data construction described in README / docs):
    1. Teacher review (``src/data/teacher_review.py``) — reject
       samples whose question, answer or key evidence is missing or
       untraceable in the plotting code;
    2. Answer verification (``src/data/answer_check.py``) — keep only
       samples whose answer is traceable to the chart;
    3. Difficulty filtering (``src/data/difficulty.py``) — keep only
       high-difficulty samples (>= ``--difficulty-threshold``).

Usage:
    python scripts/build_dataset.py --input data/raw.jsonl \
        --output data/train.jsonl --difficulty-threshold 0.7
    python scripts/build_dataset.py --demo --output /tmp/demo_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.answer_check import verify_answer  # noqa: E402
from src.data.difficulty import filter_hard  # noqa: E402
from src.data.schema import ChartQASample, EvidenceRegion  # noqa: E402
from src.data.teacher_review import review_sample  # noqa: E402

# Two deliberately hard built-in demo samples (synthetic, no real data):
# long questions, 3 evidence regions, ~16 lines of plotting code and
# 4 plotted series, so their difficulty scores clear the 0.7 default.
DEMO_CODE = """import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
plt.plot(x, [0.80, 0.85, 0.90, 0.93], label="Model-A", color="blue")
plt.plot(x, [0.70, 0.72, 0.75, 0.78], label="Model-B", color="red")
plt.plot(x, [0.65, 0.70, 0.74, 0.76], label="Model-C", color="green")
plt.plot(x, [0.60, 0.62, 0.65, 0.66], label="Model-D", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model accuracy over training epochs")
plt.legend()
plt.grid(True)
"""


def demo_samples() -> list[ChartQASample]:
    evidence = [
        EvidenceRegion(label="Model-A", code_ref="Model-A"),
        EvidenceRegion(label="Model-B", code_ref="Model-B"),
        EvidenceRegion(label="Model-C", code_ref="Model-C"),
    ]
    return [
        ChartQASample(
            sample_id="demo-hard-001",
            question=(
                "Across all four training epochs shown in the chart, which model "
                "achieves the highest accuracy at the final epoch, and by how much "
                "does it exceed the second-best model?"
            ),
            answer="Model-A",
            plotting_code=DEMO_CODE,
            evidence=list(evidence),
            source="synthetic-demo",
        ),
        ChartQASample(
            sample_id="demo-hard-002",
            question=(
                "Comparing the accuracy curves of all four models in the chart, "
                "which model shows the largest absolute accuracy gain from epoch 1 "
                "to epoch 4, and what are its start and end values?"
            ),
            answer="Model-A",
            plotting_code=DEMO_CODE,
            evidence=list(evidence),
            source="synthetic-demo",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered chart QA dataset")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input JSONL of ChartQASample records")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL of retained (hard) samples")
    parser.add_argument("--difficulty-threshold", type=float, default=0.7,
                        help="Keep samples with difficulty >= threshold (default 0.7)")
    parser.add_argument("--demo", action="store_true",
                        help="Run the pipeline on built-in demo samples")
    return parser.parse_args()


def load_samples(path: Path) -> list[ChartQASample]:
    samples: list[ChartQASample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(ChartQASample.from_dict(json.loads(line)))
    return samples


def build(
    samples: list[ChartQASample], threshold: float
) -> tuple[dict[str, int], list[ChartQASample]]:
    """Run review -> verification -> difficulty filter; return stats."""
    reviewed: list[ChartQASample] = []
    for sample in samples:
        review = review_sample(sample)
        sample.teacher_reviewed = review.verdict == "approve"
        if sample.teacher_reviewed:
            reviewed.append(sample)
    verified: list[ChartQASample] = []
    for sample in reviewed:
        sample.answer_verified = verify_answer(sample)
        if sample.answer_verified:
            verified.append(sample)
    kept = filter_hard(verified, threshold)
    return {
        "total": len(samples),
        "review_passed": len(reviewed),
        "answer_passed": len(verified),
        "retained_hard": len(kept),
    }, kept


def main() -> int:
    args = parse_args()
    if args.demo:
        samples = demo_samples()
    elif args.input is not None:
        samples = load_samples(args.input)
    else:
        print("Provide --input <samples.jsonl> or --demo.")
        return 2

    stats, kept = build(samples, args.difficulty_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for sample in kept:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    print(f"Total samples: {stats['total']}")
    print(f"Passed Teacher review: {stats['review_passed']}")
    print(f"Passed answer verification: {stats['answer_passed']}")
    print(
        f"Retained hard samples "
        f"(difficulty >= {args.difficulty_threshold}): {stats['retained_hard']}"
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
