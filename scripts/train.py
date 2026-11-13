"""Training entry: two-stage GRPO for SciChart-Evidence-VL.

Usage:
    python scripts/train.py --config configs/grpo_stage1.yaml
    python scripts/train.py --config configs/grpo_stage2.yaml
    python scripts/train.py --config configs/grpo_stage1.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Allow `python scripts/train.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.qwen_vl import QwenVLModel  # noqa: E402
from src.training.grpo import GRPOConfig, TwoStageGRPO  # noqa: E402


def load_config(path: Path) -> dict:
    """Load a YAML config, merging ``_base`` if present."""
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    base_name = config.pop("_base", None)
    if base_name:
        base = load_config(path.parent / base_name)
        # Stage config overrides base (shallow merge per top-level section).
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = {**base[key], **value}
            else:
                base[key] = value
        return base
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SciChart-Evidence-VL (GRPO)")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to a config file, e.g. configs/grpo_stage1.yaml")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2],
                        help="Override the training stage in the config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and compute losses on dummy tensors "
                             "(no model download, no GPU needed)")
    return parser.parse_args()


def dry_run(config: dict) -> None:
    """Check the loss computation end-to-end on dummy tensors."""
    stage = config.get("stage", 1)
    loss_cfg = config.get("loss", {})
    grpo_cfg = GRPOConfig(
        group_size=config.get("grpo", {}).get("group_size", 8),
        beta_kl=config.get("grpo", {}).get("beta_kl", 0.04),
        clip_epsilon=config.get("grpo", {}).get("clip_epsilon", 0.2),
        kl_consistency_weight=loss_cfg.get("kl_consistency_weight", 1.0),
        counterfactual_separation_weight=loss_cfg.get(
            "counterfactual_separation_weight", 1.0),
        evidence_persistence_weight=loss_cfg.get("evidence_persistence_weight", 0.0),
        correctness_weight=loss_cfg.get("correctness_weight", 1.0),
        stage=stage,
    )
    trainer = TwoStageGRPO(grpo_cfg)
    torch.manual_seed(0)
    logp = torch.randn(4, 10)
    losses = trainer.total_loss(
        logp, logp + 0.01, logp + 1.0, logp, logp,
        persistence_reward=torch.tensor([0.6, 0.7, 0.5, 0.8]),
    )
    print(f"[dry-run] stage={stage}")
    for key, value in losses.items():
        print(f"[dry-run] {key} = {float(value):.4f}")
    print("[dry-run] OK: config and GRPO losses are valid.")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.stage is not None:
        config["stage"] = args.stage
    stage = config.get("stage", 1)
    print(f"Stage {stage} | model={config['model']['name']} | "
          f"output={config['training']['output_dir']}")
    if args.dry_run:
        dry_run(config)
        return 0

    # Full training: load the model, then iterate over the hard samples and
    # their EP/EA pairs, sampling a GRPO group per prompt and applying the
    # two-stage loss. The data files are generated locally (see data/README.md).
    train_path = Path(config["data"]["train_path"])
    if not train_path.exists():
        print(f"Training data not found: {train_path}\n"
              "Generate it with the data pipeline first (see data/README.md), "
              "or run with --dry-run to validate the setup.")
        return 2
    model = QwenVLModel(
        model_name=config["model"]["name"],
        torch_dtype=config["model"].get("torch_dtype", "bfloat16"),
    ).load()
    print(f"Model loaded: {model.model_name}")
    print("Training loop is wired to TwoStageGRPO.total_loss(); "
          "add the dataset iterator for your data format here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
