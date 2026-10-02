"""Per-seed evaluation records and cross-seed summaries shared by both settings."""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from bon.selection import prefix_inference, select_n

ESTIMATORS = ("dm", "ipw", "dr")


def evaluate_rows(
    rows: dict[str, np.ndarray],
    grid: Sequence[int],
    truth: np.ndarray,
    rng: Callable[[int, str], np.random.Generator],
    bootstrap: int,
    alpha: float,
    gain: np.ndarray | None = None,
    bound_gain: np.ndarray | None = None,
    expected_score: np.ndarray | None = None,
) -> dict[str, list[dict]]:
    """Return calibration rows and, if ``gain`` is given, all prefix selections.

    ``gain[j]`` is the realized improvement of budget ``grid[j]`` over ``N=1``
    used for gain/harm; ``bound_gain`` is the target the Improvement-LCB claims
    to bound. Naive BoN picks the largest expected selected score.
    """
    calibration, selection = [], []
    for index, estimator in enumerate(ESTIMATORS):
        results = prefix_inference(rows[estimator], rng(index, estimator), bootstrap, alpha)
        full = results[-1]
        calibration.extend(
            dict(estimator=estimator, N=int(N), truth=float(truth[j]),
                 **{key: float(full[key][j]) for key in
                    ("estimate", "standard_error", "pointwise_ci_lower", "pointwise_ci_upper")})
            for j, N in enumerate(grid))
        if gain is None:
            continue
        for k, result in enumerate(results, 1):
            for rule, j in select_n(result).items():
                selection.append(_selection(estimator, rule, grid, k, j, gain, bound_gain,
                                            float(result["improvement_lcb"][j])))
    if gain is not None:
        for k in range(1, len(grid) + 1):
            j = int(np.argmax(expected_score[:k]))
            selection.append(_selection("reward_max", "reward_max", grid, k, j, gain,
                                        bound_gain, float("nan")))
    return dict(calibration=calibration, selection=selection)


def _selection(estimator, rule, grid, k, j, gain, bound_gain, lcb) -> dict:
    return dict(estimator=estimator, selection_rule=rule, N_max=int(grid[k - 1]),
                selected_N=int(grid[j]), gain=float(gain[j]), bound_gain=float(bound_gain[j]),
                regret=float(gain[:k].max() - gain[j]), improvement_lcb=lcb)


def write_record(path: Path, record: dict) -> None:
    """Write one per-seed record atomically so interrupted runs can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with gzip.open(temporary, "wt") as stream:
        json.dump(record, stream, allow_nan=True)
    os.replace(temporary, path)


def read_records(paths: Sequence[Path], table: str, **fields: Callable[[Path], object]) -> pd.DataFrame:
    """Concatenate one table from per-seed records, adding path-derived fields."""
    frames = []
    for path in paths:
        with gzip.open(path, "rt") as stream:
            frame = pd.DataFrame(json.load(stream)[table])
        for name, value in fields.items():
            frame[name] = value(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def calibration_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Bias, variance, RMSE and pointwise 95% coverage against the exact target."""
    x = frame.assign(
        bias=frame.estimate - frame.truth,
        ci_width=frame.pointwise_ci_upper - frame.pointwise_ci_lower,
        covered=(frame.pointwise_ci_lower - 1e-12 <= frame.truth)
        & (frame.truth <= frame.pointwise_ci_upper + 1e-12),
    )
    summary = x.groupby(keys, as_index=False).agg(
        rep_count=("bias", "size"), mean_estimate=("estimate", "mean"), truth=("truth", "mean"),
        bias=("bias", "mean"), empirical_variance=("estimate", "var"),
        mean_estimated_variance=("standard_error", lambda z: (z**2).mean()),
        empirical_sd=("estimate", "std"), rmse=("bias", lambda z: np.sqrt((z**2).mean())),
        mean_se=("standard_error", "mean"), mean_ci_width=("ci_width", "mean"),
        pointwise_coverage=("covered", "mean"))
    summary["mcse_bias"] = summary.empirical_sd / np.sqrt(summary.rep_count)
    summary["mcse_pointwise_coverage"] = np.sqrt(
        summary.pointwise_coverage * (1 - summary.pointwise_coverage) / summary.rep_count)
    return summary


def selection_summary(frame: pd.DataFrame, keys: list[str], regret: bool = False) -> pd.DataFrame:
    """Gain, harm and certificate validity of every estimator/rule/prefix.

    A certificate is a strictly positive Improvement-LCB; it is false when the
    realized gain is not positive and its bound is valid when it does not
    exceed ``bound_gain``. Naive BoN has no certificate. ``regret`` adds the
    mean gap to the best budget in the prefix and how often that gap is zero.
    """
    certifiable = frame.selection_rule.ne("reward_max")
    claim = frame.improvement_lcb > 1e-12
    x = frame.assign(
        harm=frame.gain < -1e-12,
        optimal=frame.regret <= 1e-12,
        claim=claim.where(certifiable),
        false_claim=(claim & (frame.gain <= 1e-12)).where(certifiable),
        bound_valid=(frame.improvement_lcb <= frame.bound_gain + 1e-12).where(certifiable),
    )
    summary = x.groupby(keys, as_index=False).agg(
        rep_count=("gain", "size"), mean_gain=("gain", "mean"), gain_sd=("gain", "std"),
        harm_rate=("harm", "mean"), mean_selected_N=("selected_N", "mean"),
        median_selected_N=("selected_N", "median"), certification_rate=("claim", "mean"),
        false_certificate_rate=("false_claim", "mean"),
        numerical_bound_coverage=("bound_valid", "mean"),
        **({"oracle_optimal_rate": ("optimal", "mean"), "mean_oracle_regret": ("regret", "mean")}
           if regret else {}))
    summary["mcse_harm"] = np.sqrt(summary.harm_rate * (1 - summary.harm_rate) / summary.rep_count)
    return summary


def overlap_rows(weights: np.ndarray, grid: Sequence[int]) -> list[dict]:
    """Effective sample size and weight-tail diagnostics for each budget."""
    return [dict(N=int(N), ess=float(w.sum() ** 2 / max(float(w @ w), 1e-12)),
                 weight_max=float(w.max()), weight_p99=float(np.quantile(w, 0.99)),
                 nonzero_rate=float((w > 0).mean()))
            for N, w in zip(grid, weights.T, strict=True)]


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic gzip CSV (or plain CSV for a ``.csv`` path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, compression=compression)
