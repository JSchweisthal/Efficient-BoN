"""Qwen GSM8K reference policy: metadata, sampling and non-thinking prompt rendering.

Canonical user prompt::

    Solve the following math problem step by step. At the end, clearly state
    the final numerical answer.

    {question}

The native chat template is applied with thinking disabled.
"""

from __future__ import annotations

from typing import Any

from .gemma_policy import build_prompt


MODEL_ID = "Qwen/Qwen3.5-4B"
DISPLAY_NAME = "Qwen3.5-4B"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 1536


def render_prompt(tokenizer: Any, question: str) -> tuple[str, list[int]]:
    """Render the canonical prompt with Qwen thinking explicitly disabled."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_prompt(question)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered.count("<think>") != 1 or rendered.count("</think>") != 1:
        raise ValueError("Qwen chat template changed its disabled-thinking marker")
    return rendered, tokenizer.encode(rendered, add_special_tokens=False)
