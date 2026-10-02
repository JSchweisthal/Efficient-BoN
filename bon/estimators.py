"""Paper estimators for finite-pool Best-of-N policy values.

The displayed estimators are Plug-in, BoN-IPW, and BoN-DR. For each budget
``N``, the tie-aware rank weight estimates ``pi_N(a|x) / pi_b(a|x)`` from an
independent behavior pool. BoN-DR uses the row contribution
``plugin_N + weight_N * (Y - mu_hat(X,A))``. All budgets share rows so paired
uncertainty and budget selection use the same observations. Scores closer than
a relative 1e-12 are treated as ties, broken uniformly at random.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


TORCH_DTYPE = torch.float64


def _log_comb(n: torch.Tensor | float | int, k: torch.Tensor | float | int) -> torch.Tensor:
    """Return log binomial coefficients and negative infinity outside support."""
    nt = torch.as_tensor(n, dtype=TORCH_DTYPE)
    kt = torch.as_tensor(k, dtype=TORCH_DTYPE, device=nt.device)
    valid = (kt >= 0) & (kt <= nt)
    value = torch.lgamma(nt + 1) - torch.lgamma(kt + 1) - torch.lgamma(nt - kt + 1)
    return torch.where(valid, value, torch.full_like(value, -torch.inf))


def _quantize(values: torch.Tensor) -> torch.Tensor:
    """Round each row of scores to a relative 1e-12 grid so near-ties become ties."""
    scale = 1e-12 * torch.maximum(
        torch.ones((values.shape[0], 1), dtype=TORCH_DTYPE, device=values.device),
        values.abs().max(1, keepdim=True).values,
    )
    return torch.round(values / scale)


@dataclass(frozen=True)
class ScalarScoreComponents:
    """Per-observation BoN-DR, Plug-in, and BoN-IPW rank-weight rows."""

    dr: torch.Tensor
    plugin: torch.Tensor
    ratio: torch.Tensor


def finite_pool_bon_ratio_rows(
    pool_counts: torch.Tensor,
    logged_actions: torch.Tensor,
    scores: torch.Tensor,
    N_grid: Sequence[int],
) -> torch.Tensor:
    """Vectorize tie-aware finite-pool BoN rank weights over rows and budgets."""
    if pool_counts.ndim != 2 or logged_actions.ndim != 1:
        raise ValueError("pool_counts and logged_actions must have rank 2 and 1")
    n, n_actions = pool_counts.shape
    if logged_actions.shape != (n,):
        raise ValueError("logged_actions must contain one action per pool")
    if scores.ndim == 1:
        scores = scores.expand(n, -1)
    if scores.shape != (n, n_actions):
        raise ValueError("scores must match pool_counts")
    pool_sizes = pool_counts.sum(1)
    if n == 0 or not torch.all(pool_sizes == pool_sizes[0]):
        raise ValueError("all nonempty pools must have the same size")
    pool_size = int(pool_sizes[0].item())
    budgets = tuple(int(value) for value in N_grid)
    if not budgets or budgets[0] < 1 or max(budgets) > pool_size + 1:
        raise ValueError("candidate budgets must be between 1 and pool size + 1")

    values = scores.to(TORCH_DTYPE)
    counts = pool_counts.to(dtype=TORCH_DTYPE, device=values.device)
    actions = logged_actions.to(device=values.device)
    levels = _quantize(values)
    logged_levels = levels.gather(1, actions[:, None])
    lower = (counts * (levels < logged_levels)).sum(1)
    equal = (counts * (levels == logged_levels)).sum(1)
    ratios = torch.empty((n, len(budgets)), dtype=TORCH_DTYPE, device=values.device)
    for column, budget in enumerate(budgets):
        if budget == 1:
            ratios[:, column] = 1.0
            continue
        tie_draws = torch.arange(
            min(int(equal.max().item()), budget - 1) + 1,
            dtype=TORCH_DTYPE,
            device=values.device,
        )
        log_terms = (
            _log_comb(equal[:, None], tie_draws)
            + _log_comb(lower[:, None], budget - 1 - tie_draws)
            - _log_comb(pool_size, budget - 1)
            - torch.log(tie_draws + 1.0)
        )
        ratios[:, column] = (
            budget * torch.exp(log_terms).sum(1)
        ).clamp(0.0, float(budget))
    return ratios


def finite_pool_bon_ratio_table(scores: torch.Tensor, N_grid: Sequence[int]) -> torch.Tensor:
    """Return exact BoN ratios for every possible uniform logged pool slot."""
    if scores.ndim != 2 or scores.shape[0] < 1 or scores.shape[1] < 1:
        raise ValueError("scores must be a nonempty context-by-slot table")
    n, pool_size = scores.shape
    budgets = tuple(int(value) for value in N_grid)
    if (
        not budgets
        or budgets != tuple(sorted(set(budgets)))
        or budgets[0] < 1
        or budgets[-1] > pool_size
    ):
        raise ValueError("budgets must be sorted, unique, and within the pool")
    values = scores.to(TORCH_DTYPE)
    quantized = _quantize(values)
    order = torch.argsort(quantized, dim=1, stable=True)
    sorted_levels = quantized.gather(1, order)
    starts = torch.ones((n, pool_size), dtype=torch.bool, device=values.device)
    starts[:, 1:] = sorted_levels[:, 1:] != sorted_levels[:, :-1]
    local_groups = starts.cumsum(1) - 1
    row_offsets = torch.arange(n, device=values.device)[:, None] * pool_size
    flat_groups = (local_groups + row_offsets).flatten()
    group_sizes = torch.zeros(
        n * pool_size, dtype=TORCH_DTYPE, device=values.device
    )
    group_sizes.scatter_add_(
        0,
        flat_groups,
        torch.ones(n * pool_size, dtype=TORCH_DTYPE, device=values.device),
    )
    group_sizes = group_sizes.view(n, pool_size)
    upper = group_sizes.cumsum(1)
    lower = upper - group_sizes
    sorted_ratios = torch.empty(
        (n, pool_size, len(budgets)), dtype=TORCH_DTYPE, device=values.device
    )
    for column, budget in enumerate(budgets):
        if budget == 1:
            level_ratio = torch.ones_like(group_sizes)
        else:
            denominator = _log_comb(pool_size - 1, budget - 1)
            probability_mass = torch.exp(
                _log_comb(upper, budget) - denominator
            ) - torch.exp(_log_comb(lower, budget) - denominator)
            level_ratio = budget * probability_mass / group_sizes.clamp_min(1.0)
        sorted_ratios[:, :, column] = level_ratio.gather(1, local_groups)
    inverse = torch.empty_like(order)
    inverse.scatter_(
        1,
        order,
        torch.arange(pool_size, device=values.device)[None, :].expand(n, -1),
    )
    result = sorted_ratios.gather(
        1, inverse[:, :, None].expand(-1, -1, len(budgets))
    )
    upper_bounds = torch.as_tensor(budgets, dtype=TORCH_DTYPE, device=values.device)
    if torch.any(result < -1e-10) or torch.any(result > upper_bounds + 1e-8):
        raise AssertionError("density-ratio table violates [0,N] bounds")
    return result.clamp_min(0.0)


def finite_pool_bon_plugin_rows(
    full_pool_counts: torch.Tensor,
    scores: torch.Tensor,
    mu_hat_by_obs: torch.Tensor,
    N_grid: Sequence[int],
) -> torch.Tensor:
    """Vectorize tie-aware finite-pool Plug-in values over rows and budgets."""
    if full_pool_counts.ndim != 2:
        raise ValueError("full_pool_counts must have rank 2")
    n, n_actions = full_pool_counts.shape
    if scores.shape != (n, n_actions) or mu_hat_by_obs.shape != (n, n_actions):
        raise ValueError("scores and nuisance tables must match full_pool_counts")
    budgets = tuple(int(value) for value in N_grid)
    totals = full_pool_counts.sum(1)
    if n == 0 or not budgets or budgets[0] < 1:
        raise ValueError("rows and candidate budgets must be nonempty")
    if torch.any(totals < max(budgets)):
        raise ValueError("candidate budget exceeds a full pool size")

    values = scores.to(TORCH_DTYPE)
    counts = full_pool_counts.to(dtype=TORCH_DTYPE, device=values.device)
    nuisance = mu_hat_by_obs.to(dtype=TORCH_DTYPE, device=values.device)
    quantized = _quantize(values)
    order = torch.argsort(quantized, dim=1, stable=True)
    sorted_levels = quantized.gather(1, order)
    sorted_counts = counts.gather(1, order)
    sorted_weighted_mu = (counts * nuisance).gather(1, order)
    starts = torch.ones((n, n_actions), dtype=torch.bool, device=values.device)
    starts[:, 1:] = sorted_levels[:, 1:] != sorted_levels[:, :-1]
    local_groups = starts.cumsum(1) - 1
    row_offsets = torch.arange(n, device=values.device)[:, None] * n_actions
    flat_groups = (local_groups + row_offsets).flatten()
    group_mass = torch.zeros(n * n_actions, dtype=TORCH_DTYPE, device=values.device)
    group_mu_mass = torch.zeros_like(group_mass)
    group_mass.scatter_add_(0, flat_groups, sorted_counts.flatten())
    group_mu_mass.scatter_add_(0, flat_groups, sorted_weighted_mu.flatten())
    group_mass = group_mass.view(n, n_actions)
    group_mu_mass = group_mu_mass.view(n, n_actions)
    group_mean = torch.where(
        group_mass > 0,
        group_mu_mass / group_mass.clamp_min(1.0),
        torch.zeros_like(group_mass),
    )
    upper = group_mass.cumsum(1)
    lower = upper - group_mass
    totals_float = totals.to(dtype=TORCH_DTYPE, device=values.device)[:, None]
    plugin = torch.empty((n, len(budgets)), dtype=TORCH_DTYPE, device=values.device)
    for column, budget in enumerate(budgets):
        budget_tensor = torch.full_like(upper, float(budget))
        denominator = _log_comb(totals_float, float(budget))
        weights = torch.exp(
            _log_comb(upper, budget_tensor) - denominator
        ) - torch.exp(_log_comb(lower, budget_tensor) - denominator)
        plugin[:, column] = (weights * group_mean).sum(1)
        if not torch.allclose(
            weights.sum(1),
            torch.ones(n, dtype=TORCH_DTYPE, device=values.device),
            atol=1e-8,
        ):
            raise AssertionError("vectorized plug-in weights do not sum to one")
    return plugin


def finite_pool_scalar_score_components(
    actions: torch.Tensor,
    outcomes: torch.Tensor,
    pool_counts: torch.Tensor,
    scores: torch.Tensor,
    mu_hat_by_obs: torch.Tensor,
    N_grid: Sequence[int],
    *,
    plugin: torch.Tensor | None = None,
) -> ScalarScoreComponents:
    """Compute scalar DR rows with vectorized ratios and cached Plug-in rows."""
    n = actions.numel()
    if pool_counts.shape != scores.shape or scores.shape != mu_hat_by_obs.shape:
        raise ValueError("pool, score, and nuisance tables must be aligned")
    if outcomes.shape != (n,) or scores.shape[0] != n:
        raise ValueError("actions and outcomes must contain one value per row")
    ratios = finite_pool_bon_ratio_rows(pool_counts, actions, scores, N_grid)
    if plugin is None:
        full_counts = pool_counts.clone()
        full_counts.scatter_add_(
            1,
            actions[:, None],
            torch.ones_like(actions[:, None], dtype=full_counts.dtype),
        )
        plugin = finite_pool_bon_plugin_rows(
            full_counts, scores, mu_hat_by_obs, N_grid
        )
    expected = (n, len(N_grid))
    if plugin.shape != expected or plugin.device != scores.device:
        raise ValueError("cached Plug-in rows have the wrong shape or device")
    row_ids = torch.arange(n, device=actions.device)
    logged_mu = mu_hat_by_obs[row_ids, actions]
    dr = plugin + ratios * (outcomes - logged_mu).unsqueeze(1)
    return ScalarScoreComponents(dr=dr, plugin=plugin, ratio=ratios)
