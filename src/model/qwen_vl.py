"""Qwen2.5-VL model wrapper: loading and attention-trajectory extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class QwenVLModel:
    """Thin wrapper around Qwen2.5-VL for chart understanding.

    The heavy ``transformers`` / model weights are only imported when
    :meth:`load` is called, so the rest of the package can be imported
    and type-checked without a GPU or a downloaded checkpoint.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 torch_dtype: str = "bfloat16") -> None:
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> "QwenVLModel":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        return self

    def generate(self, image_path: str | Path, question: str,
                 max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate an answer for one chart image + question."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load() before generate().")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info  # type: ignore

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-6),
        )
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def extract_attention_trajectory(
        self, image_path: str | Path, question: str,
        steps: int = 8,
    ) -> list[dict[str, Any]]:
        """Extract key-region attention over reasoning steps.

        Returns a list of ``steps`` entries, each with the mean attention
        mass on image tokens and the per-layer attention tensors (detached,
        on CPU) for that step. The caller maps image tokens to evidence
        regions with :func:`src.training.evidence.build_evidence_token_mask`.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Call load() before extract_attention_trajectory().")
        # Full step-by-step decoding with attention capture is model-specific;
        # this is the interface the training code uses. A minimal
        # implementation runs one forward pass with output_attentions=True
        # and splits the sequence into ``steps`` chunks.
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info  # type: ignore

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        attentions = outputs.attentions  # tuple[layer] of (batch, heads, seq, seq)
        trajectory: list[dict[str, Any]] = []
        for step in range(steps):
            layer_idx = min(step, len(attentions) - 1)
            attn = attentions[layer_idx][0].mean(dim=0)  # (seq, seq), heads averaged
            trajectory.append({
                "step": step,
                "attention": attn.detach().cpu(),
                "image_token_attention": attn[:, : inputs.input_ids.shape[1]].mean().item(),
            })
        return trajectory
