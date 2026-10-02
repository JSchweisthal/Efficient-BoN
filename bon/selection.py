"""Paired uncertainty and budget selection on shared estimator rows.

Rows are an ``n x J`` matrix whose column means estimate ``V(pi_N)`` on a
sorted budget grid starting at ``N=1``. Improvement-LCB uses a one-sided
simultaneous lower bound for ``V(pi_N) - V(pi_1)`` from a Gaussian multiplier
bootstrap, recalibrated on every prefix ``N <= N_max`` of the grid.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def prefix_inference(
    rows: np.ndarray, rng: np.random.Generator, repetitions: int = 2000, alpha: float = 0.05
) -> list[dict]:
    """Return pointwise intervals and paired-improvement LCBs for every grid prefix.

    Conditional on the rows, Gaussian multipliers have covariance
    ``centered.T @ centered / n``; drawing them in ``J`` dimensions preserves
    that distribution and makes all prefix maxima cheap.
    """
    rows = np.asarray(rows, dtype=np.float64)
    n, width = rows.shape
    if n < 2 or not np.isfinite(rows).all():
        raise ValueError("at least two finite observation rows are required")
    mean = rows.mean(0)
    centered = rows - mean
    covariance = centered.T @ centered / n
    eig, vectors = np.linalg.eigh(covariance)
    draws = rng.normal(size=(repetitions, width)) @ (vectors * np.sqrt(eig.clip(0))).T
    sigma = np.sqrt(np.diag(covariance).clip(0))
    delta = mean - mean[0]
    dsigma = (rows - rows[:, :1]).std(0)
    dz = (draws - draws[:, :1]) / np.maximum(dsigma, 1e-12)
    dse = dsigma / np.sqrt(n)
    # Pointwise Wald intervals use the sample variance (ddof=1).
    point_se = sigma / np.sqrt(n) * np.sqrt(n / (n - 1))
    half_width = norm.ppf(1 - alpha / 2) * point_se
    results = []
    for k in range(1, width + 1):
        critical = float(np.quantile(dz[:, 1:k].max(1), 1 - alpha)) if k > 1 else 0.0
        results.append(dict(
            estimate=mean[:k],
            standard_error=point_se[:k],
            pointwise_ci_lower=(mean - half_width)[:k],
            pointwise_ci_upper=(mean + half_width)[:k],
            improvement_lcb=(delta - critical * dse)[:k],
            critical_improvement=critical,
        ))
    return results


def select_n(result: dict) -> dict[str, int]:
    """Return grid indices chosen by Value-max and Improvement-LCB.

    ``np.argmax`` picks the smallest budget among ties. The improvement bound
    at ``N=1`` is exactly zero, so Improvement-LCB falls back to ``N=1`` unless
    some budget has a strictly positive certified improvement.
    """
    return {
        "value": int(np.argmax(result["estimate"])),
        "improvement_lcb": int(np.argmax(result["improvement_lcb"])),
    }
