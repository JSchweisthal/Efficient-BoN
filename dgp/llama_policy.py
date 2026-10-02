"""Llama GSM8K reference policy: metadata, sampling and prompt rendering.

Canonical user prompt::

    Solve the following math problem step by step. At the end, clearly state
    the final numerical answer.

    {question}

The native chat template is applied with the frozen generation date.
"""

from __future__ import annotations

from typing import Any

from .gemma_policy import build_prompt


MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DISPLAY_NAME = "Llama-3.2-3B-Instruct"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 768


def render_prompt(tokenizer: Any, question: str) -> tuple[str, list[int]]:
    """Render the canonical user-only prompt with the frozen chat-template date."""
    prompt = build_prompt(question)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        date_string="10 Sep 2026",
    )
    if prompt not in rendered:
        raise ValueError("Llama chat template changed the user prompt")
    return rendered, tokenizer.encode(rendered, add_special_tokens=False)
