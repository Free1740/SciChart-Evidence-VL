"""Evaluation entry: run the 8 public benchmarks and write a report.

Usage:
    python scripts/evaluate.py --config configs/eval.yaml
    python scripts/evaluate.py --config configs/eval.yaml --data-dir data/eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.evaluate import BENCHMARKS, run_evaluation  # noqa: E402
from src.model.qwen_vl import QwenVLModel  # noqa: E402


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SciChart-Evidence-VL")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory with <Benchmark>.jsonl files")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override the report output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    data_dir = args.data_dir or Path("data/eval")
    output = args.output or Path(config["output"]["report_path"])
    benchmarks = config.get("benchmarks", BENCHMARKS)
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Data dir: {data_dir} | report: {output}")

    model = QwenVLModel(
        model_name=config["model"]["name"],
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
    )
    try:
        model.load()
    except Exception as exc:  # noqa: BLE001 - skeleton must stay runnable
        print(f"Warning: could not load model ({exc}); "
              "running with a dummy predictor (scores will be 0).")
        model = None

    def predict(image_path: str, question: str) -> str:
        if model is None:
            return ""
        return model.generate(
            image_path, question,
            max_new_tokens=config.get("generation", {}).get("max_new_tokens", 512),
            temperature=config.get("generation", {}).get("temperature", 0.0),
        )

    report = run_evaluation(benchmarks, predict, data_dir, output)
    print(f"Average: {report['average']:.2f} "
          f"(baseline {report['baseline_average']}, "
          f"reported {report['reported_average']})")
    print(f"Report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
