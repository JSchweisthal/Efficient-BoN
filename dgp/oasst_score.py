"""OASST reward scoring and Parquet score-store compatibility.

The scorer loads an already-present local checkpoint only. Stored scores use
the ``question_id``/``sample_id`` keys and a scalar ``score`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn


MODEL_ID = "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5"
DISPLAY_NAME = "OASST-RM-2.1-Pythia-1.4B"
FORMAT_TEMPLATE = "<|prompter|>{prompt}<|endoftext|><|assistant|>{response}<|endoftext|>"


def _register_oasst_architecture() -> None:
    """Register the checkpoint's local GPT-NeoX scalar-head architecture."""
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from transformers.models.gpt_neox.modeling_gpt_neox import (
        GPTNeoXConfig,
        GPTNeoXModel,
        GPTNeoXPreTrainedModel,
    )
    from transformers.utils import ModelOutput

    class RewardConfig(GPTNeoXConfig):
        model_type = "gpt_neox_reward_model"

        def __init__(self, pooling: str = "last", **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.pooling = pooling or "last"

    @dataclass
    class RewardOutput(ModelOutput):
        logits: Optional[torch.FloatTensor] = None

    class RewardModel(GPTNeoXPreTrainedModel):
        config_class = RewardConfig

        def __init__(self, config: GPTNeoXConfig) -> None:
            if type(config) is GPTNeoXConfig:
                config = RewardConfig.from_dict(config.to_dict())
            super().__init__(config)
            self.gpt_neox = GPTNeoXModel(config)
            self.out_proj = nn.Linear(config.hidden_size, 1)
            self.pooling = config.pooling
            self.post_init()

        def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> RewardOutput:
            hidden = self.gpt_neox(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
            ).last_hidden_state
            if self.pooling == "mean":
                weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            else:
                # Index of the last attended token under left or right padding.
                last = attention_mask.cumsum(1).argmax(1)
                pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
            return RewardOutput(logits=self.out_proj(pooled))

    try:
        AutoConfig.register(RewardConfig.model_type, RewardConfig)
    except ValueError:
        pass
    try:
        AutoModelForSequenceClassification.register(RewardConfig, RewardModel)
    except ValueError:
        pass


def format_input(prompt: str, response: str) -> str:
    """Format one user/assistant pair exactly as in the stored OASST scores."""
    if not prompt.strip() or not response.strip():
        raise ValueError("prompt and response must be nonempty")
    return FORMAT_TEMPLATE.format(prompt=prompt, response=response)


class OASSTScorer:
    """Score response pairs with an existing local OASST checkpoint."""

    def __init__(self, model_path: Path, device: str = "cuda:0") -> None:
        """Load local files without downloading or modifying a model cache."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _register_oasst_architecture()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            raise ValueError("OASST tokenizer has no padding token")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16
        ).to(device).eval()
        self.max_length = min(
            int(self.model.config.max_position_embeddings), int(self.tokenizer.model_max_length)
        )

    def score(self, prompts: Sequence[str], responses: Sequence[str]) -> torch.Tensor:
        """Return one raw scalar logit for each aligned prompt/response pair."""
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must align")
        texts = [format_input(p, r) for p, r in zip(prompts, responses, strict=True)]
        # Inputs longer than the context window keep their first max_length tokens.
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = self.model(**{key: value.to(self.device) for key, value in tokens.items()}).logits
        if logits.shape != (len(texts), 1):
            raise RuntimeError("unexpected OASST reward shape")
        if not torch.isfinite(logits).all():
            raise RuntimeError("OASST produced non-finite rewards")
        return logits[:, 0].float().cpu()
