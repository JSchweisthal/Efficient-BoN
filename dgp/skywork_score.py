"""Skywork reward scoring using an existing local checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


MODEL_ID = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
DISPLAY_NAME = "Skywork-Reward-V2-Qwen3-1.7B"


class SkyworkScorer:
    """Score user/assistant pairs with the checkpoint's native chat template."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda:0",
        max_length: int = 16384,
        batch_size: int = 32,
    ) -> None:
        """Load local files only; no download or cache mutation is attempted."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            num_labels=1,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def score(self, prompt: str, responses: Sequence[str]) -> np.ndarray:
        """Return raw float32 logits and fail instead of silently truncating."""
        output: list[float] = []
        for start in range(0, len(responses), self.batch_size):
            batch = responses[start : start + self.batch_size]
            texts = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for response in batch
            ]
            tokens = self.tokenizer(texts, padding=True, return_tensors="pt", truncation=False)
            if tokens["input_ids"].shape[1] > self.max_length:
                raise ValueError("input exceeds the declared Skywork context window")
            with torch.inference_mode():
                logits = self.model(
                    **{key: value.to(self.device) for key, value in tokens.items()}
                ).logits
            if logits.shape != (len(batch), 1) or not torch.isfinite(logits).all():
                raise RuntimeError("invalid Skywork reward logits")
            output.extend(logits[:, 0].float().cpu().tolist())
        return np.asarray(output, dtype=np.float32)
