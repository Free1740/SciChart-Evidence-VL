"""Build EP/EA counterfactual pairs from chart QA samples.

Usage:
    python scripts/build_counterfactual.py --input data/train.jsonl --output data/counterfactual_pairs.jsonl
    python scripts/build_counterfactual.py --demo --output data/counterfactual_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.counterfactual import EXAMPLE_CODE, CounterfactualBuilder  # noqa: E402
from src.data.schema import ChartQASample, EvidenceRegion  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EP/EA counterfactual pairs")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input JSONL of ChartQASample records")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL of CounterfactualPair records")
    parser.add_argument("--demo", action="store_true",
                        help="Build one demo pair from the built-in example code")
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
    if args.demo:
        samples = [ChartQASample(
            sample_id="demo-001",
            question="Which model has higher accuracy at x=3?",
            answer="Model-A",
            plotting_code=EXAMPLE_CODE,
            evidence=[EvidenceRegion(label="Model-A", code_ref="Model-A")],
        )]
    elif args.input is not None:
        samples = load_samples(args.input)
    else:
        print("Provide --input <samples.jsonl> or --demo.")
        return 2

    builder = CounterfactualBuilder()
    pairs = builder.build_pairs(samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
    print(f"Built {len(pairs)} counterfactual pairs from {len(samples)} samples "
          f"(skipped {len(samples) - len(pairs)} with unlocatable evidence).")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
