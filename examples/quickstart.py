"""Quickstart: difficulty scoring + EP/EA counterfactuals on the examples.

Usage:
    python examples/quickstart.py
    python examples/quickstart.py --input examples/sample_qa.jsonl

Reads synthetic example samples, scores their difficulty, builds an
EP/EA counterfactual pair for each, and prints whether the EP view
keeps the key evidence and the EA view removes it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.counterfactual import CounterfactualBuilder  # noqa: E402
from src.data.difficulty import score_difficulty  # noqa: E402
from src.data.schema import ChartQASample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickstart on example samples")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples" / "sample_qa.jsonl",
        help="Input JSONL of ChartQASample records",
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[ChartQASample]:
    samples: list[ChartQASample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(ChartQASample.from_dict(json.loads(line)))
    return samples


def main() -> int:
    args = parse_args()
    samples = load_samples(args.input)
    builder = CounterfactualBuilder()
    print(f"Loaded {len(samples)} example sample(s) from {args.input}")
    for sample in samples:
        difficulty = score_difficulty(sample)
        pair = builder.build_pair(sample)
        evidence = sample.evidence[0].label if sample.evidence else "?"
        ep_keeps = evidence in pair.ep_plotting_code
        ea_removes = evidence not in pair.ea_plotting_code
        print(f"\n[{sample.sample_id}] {sample.question}")
        print(f"  Answer: {sample.answer} | difficulty: {difficulty:.2f}")
        print(f"  EP keeps {evidence!r}: {ep_keeps}")
        print(f"  EA removes {evidence!r}: {ea_removes}")
        print(f"  Ablation: {pair.ablation_description}")
        if not (ep_keeps and ea_removes):
            print("  WARNING: counterfactual check failed for this sample.")
            return 1
    print("\nAll example counterfactual checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
