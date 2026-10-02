"""Controlled rare-tail DGP used by every synthetic paper experiment.

The behavior policy places four percent mass on eight low-quality tail actions.
Score error changes only the induced BoN policy; reward-function error changes
only the fitted nuisance. This separation is the source of Figures 2, 5--7,
and 12 and Tables 1, 4, and 6.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


DTYPE = torch.float64


def _generator(seed: int, stream: str) -> torch.Generator:
    """Return the named deterministic Torch stream used by the frozen study."""
    digest = hashlib.blake2b(
        f"rare_tail_v3\0{seed}\0{stream}".encode(), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "little") & ((1 << 63) - 1)
    return torch.Generator().manual_seed(value)


@dataclass(frozen=True)
class RareTailWorld:
    """Frozen behavior, truth, score, and reward-function tables."""

    probabilities: torch.Tensor
    quality: torch.Tensor
    scores: torch.Tensor
    nuisance: torch.Tensor
    tail: torch.Tensor


def make_rare_tail(
    score_error: float = 0.65, reward_function_error: float = 0.4
) -> RareTailWorld:
    """Construct the paper's two-context, 24-action synthetic population."""
    quality = torch.cat(
        (
            torch.linspace(0.55, 0.85, 16, dtype=DTYPE),
            torch.linspace(0.05, 0.25, 8, dtype=DTYPE),
        )
    )
    quality = quality[None, :] + torch.tensor([[-0.04], [0.04]], dtype=DTYPE)
    tail = torch.arange(24) >= 16
    probabilities = torch.where(tail, 0.04 / 8, 0.96 / 16).to(DTYPE)
    probabilities /= probabilities.sum()
    return RareTailWorld(
        probabilities.expand(2, -1).clone(),
        quality,
        quality + score_error * tail,
        (quality + reward_function_error * tail).clamp(0, 1),
        tail,
    )


def sample_rare_tail(
    world: RareTailWorld, n: int, pool_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw logged outcomes and independent auxiliary behavior-pool counts."""
    if n < 2 or pool_size < 1:
        raise ValueError("n >= 2 and pool_size >= 1 are required")
    contexts = torch.randint(2, (n,), generator=_generator(seed, "contexts"))
    actions = torch.multinomial(
        world.probabilities[contexts], 1, generator=_generator(seed, "logged")
    ).squeeze(1)
    outcomes = torch.bernoulli(
        world.quality[contexts, actions], generator=_generator(seed, "outcomes")
    )
    remaining = torch.full((n,), float(pool_size), dtype=DTYPE)
    counts = torch.zeros((n, 24), dtype=torch.int64)
    generator = _generator(seed, f"auxiliary_M{pool_size}")
    probabilities = world.probabilities[0]
    for action in range(23):
        draw = torch.binomial(
            remaining,
            (probabilities[action] / probabilities[action:].sum()).expand(n),
            generator=generator,
        )
        counts[:, action] = draw.long()
        remaining -= draw
    counts[:, -1] = remaining.long()
    return contexts, actions, outcomes, counts


def true_bon_policy(probabilities: torch.Tensor, scores: torch.Tensor, N: int) -> torch.Tensor:
    """Return the exact BoN policy under uniform maximal-slot tie breaking."""
    if N < 1:
        raise ValueError("N must be positive")
    if scores.ndim == 1:
        scores = scores.expand_as(probabilities)
    output = torch.zeros_like(probabilities, dtype=DTYPE)
    for context in range(probabilities.shape[0]):
        scale = 1e-12 * torch.maximum(
            torch.ones((), dtype=DTYPE), scores[context].abs().max()
        )
        levels = torch.unique(
            torch.round(scores[context] / scale), sorted=True, return_inverse=True
        )[1]
        masses = torch.zeros(int(levels.max()) + 1, dtype=DTYPE)
        masses.scatter_add_(0, levels, probabilities[context])
        upper = masses.cumsum(0).clamp(max=1)
        relative = (masses / upper).clamp(0, 1)
        geometric = torch.where(
            relative > 0,
            -torch.expm1(N * torch.log1p(-relative)) / relative,
            torch.full_like(relative, float(N)),
        )
        output[context] = probabilities[context] * (
            upper.pow(N - 1) * geometric
        )[levels]
    if not torch.allclose(output.sum(1), torch.ones(output.shape[0], dtype=DTYPE), atol=1e-9):
        raise AssertionError("invalid exact BoN policy")
    return output


def rare_tail_oracle(world: RareTailWorld, grid: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """Return exact value, score, tail mass, and efficient variance by budget."""
    policies = torch.stack(
        [true_bon_policy(world.probabilities, world.scores, budget) for budget in grid],
        dim=2,
    )
    values = (policies * world.quality[:, :, None]).sum(1)
    residual_variance = (
        policies.square()
        / world.probabilities[:, :, None]
        * (world.quality * (1 - world.quality))[:, :, None]
    ).sum(1).mean(0)
    return {
        "truth": values.mean(0),
        "expected_score": (policies * world.scores[:, :, None]).sum(1).mean(0),
        "tail_mass": policies[:, world.tail].sum(1).mean(0),
        "efficient_variance": values.var(0, correction=0) + residual_variance,
    }
