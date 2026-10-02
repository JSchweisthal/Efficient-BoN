"""Gemma GSM8K reference policy and shared response generation.

Canonical user prompt::

    Solve the following math problem step by step. At the end, clearly state
    the final numerical answer.

    {question}

Responses are keyed by ``question_id``/``sample_id``. Generation is only run
to build response banks; experiments consume stored scores and outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


MODEL_ID = "google/gemma-2-2b-it"
DISPLAY_NAME = "Gemma-2-2B-IT"
TEMPERATURE = 1.0
TOP_P = 1.0
GSM8K_PROMPT = (
    "Solve the following math problem step by step. "
    "At the end, clearly state the final numerical answer."
)
RESPONSE_COLUMNS = (
    "question_id",
    "sample_id",
    "response",
    "num_tokens",
    "finish_reason",
    "shard_seed",
)


@dataclass(frozen=True)
class GeneratedResponse:
    """One response row compatible with the existing Parquet schema."""

    question_id: int
    sample_id: int
    response: str
    num_tokens: int
    finish_reason: str
    shard_seed: int


def build_prompt(question: str) -> str:
    """Return the canonical zero-shot chain-of-thought prompt."""
    question = question.strip()
    if not question:
        raise ValueError("question must be nonempty")
    return f"{GSM8K_PROMPT}\n\n{question}"


def generate_local(
    *,
    model_path: Path,
    question: str,
    question_id: int,
    count: int,
    seed: int,
    max_new_tokens: int,
    render: Callable[[Any, str], tuple[str, list[int]]] | None = None,
    device: str = "cuda:0",
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
) -> list[GeneratedResponse]:
    """Generate one response shard from a local checkpoint.

    ``render`` supplies another policy's chat-template prompt (see
    ``llama_policy``/``qwen_policy``) together with its sampling parameters.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for response generation")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16
    ).to(device).eval()
    if render is None:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(question)}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
    else:
        encoded = torch.tensor([render(tokenizer, question)[1]], dtype=torch.long)
    encoded = encoded.to(device).repeat(count, 1)
    # Gemma ends a turn with <end_of_turn>, which is not tokenizer.eos_token_id.
    configured = model.generation_config.eos_token_id
    stop_ids = set(configured if isinstance(configured, list) else [configured])
    stop_ids.add(tokenizer.eos_token_id)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        outputs = model.generate(
            encoded,
            attention_mask=torch.ones_like(encoded),
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            max_new_tokens=max_new_tokens,
        )[:, encoded.shape[1]:]
    records = []
    for sample_id, tokens in enumerate(outputs.tolist()):
        stop = next((i for i, token in enumerate(tokens) if token in stop_ids), None)
        length = len(tokens) if stop is None else stop + 1
        response = tokenizer.decode(tokens[:length], skip_special_tokens=True).strip()
        if not response:
            raise RuntimeError("model generated an empty response")
        records.append(GeneratedResponse(
            question_id, sample_id, response, length,
            "length" if stop is None else "eos", seed
        ))
    return records
