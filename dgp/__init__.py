"""Synthetic and GSM8K data interfaces used by the paper experiments."""

from .gemma_policy import GSM8K_PROMPT, build_prompt
from .score_artifacts import load_score_artifacts
from .synthetic_dgp import RareTailWorld, make_rare_tail, rare_tail_oracle, sample_rare_tail

__all__ = [
    "GSM8K_PROMPT",
    "RareTailWorld",
    "build_prompt",
    "load_score_artifacts",
    "make_rare_tail",
    "rare_tail_oracle",
    "sample_rare_tail",
]
