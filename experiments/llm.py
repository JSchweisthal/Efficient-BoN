"""GSM8K evaluation from stored reward scores and correctness labels.

No responses are generated and no reward model is run. Each split (train,
validation, test) supplies a ``question x sample`` score table and outcome
table. The outcome regression ``mu_hat(s)`` is a cubic-spline logistic fit on
one random training response per prompt. Two designs are evaluated:

* ``fixed_bank``: one logged response per validation prompt, the remaining
  bank as the auxiliary pool; budgets are selected on validation and scored by
  their exact value on the held-out test banks.
* ``iid_empirical``: prompts, logged and ``M`` auxiliary responses are drawn
  with replacement; calibration targets the exact value on the full empirical
  validation population.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from bon.estimators import (
    finite_pool_bon_plugin_rows, finite_pool_bon_ratio_table, finite_pool_scalar_score_components,
)
from dgp import oasst_score, skywork_score
from dgp.score_artifacts import load_score_artifacts
from dgp.synthetic_dgp import true_bon_policy

from .common import (
    calibration_summary, evaluate_rows, overlap_rows, read_records, selection_summary, write_csv,
    write_record,
)

GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
SEED = 20260916
REPETITIONS = dict(fixed_bank=100, iid_empirical=500)
AUXILIARY_DRAWS = 512
BOOTSTRAP = 2000
ALPHA = 0.05
REWARD_MODEL_IDS = dict(oasst=oasst_score.MODEL_ID, skywork=skywork_score.MODEL_ID)
SPLINE = "score_spline"


def rng(*parts) -> np.random.Generator:
    """Return an independent named stream; data draws are shared by estimators."""
    digest = hashlib.sha256(repr((SEED, parts)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def load_table(root: Path, column: str, model_id: str | None = None) -> np.ndarray:
    """Load a complete ``question x sample`` table sorted by question and sample id."""
    if column == "score":
        frame = load_score_artifacts(root, expected_model_id=model_id)
    else:
        paths = sorted((root / "outcomes").glob("q*.parquet")) or sorted(root.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no outcome Parquet files below {root}")
        frame = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)
    table = frame.pivot(index="question_id", columns="sample_id", values=column).sort_index()
    if table.isna().any().any():
        raise ValueError(f"{root} does not cover every question/sample pair")
    return table.to_numpy(dtype=np.float64)


def fit_score_spline(scores: np.ndarray, outcomes: np.ndarray):
    """Fit ``P(Y=1 | s)`` with six quantile knots and a fixed L2 penalty."""
    model = make_pipeline(
        SplineTransformer(n_knots=6, degree=3, knots="quantile", extrapolation="constant",
                          include_bias=False),
        LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
    )
    return model.fit(scores.reshape(-1, 1), outcomes.reshape(-1))


def exact_rows(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Exact per-prompt BoN value of ``values`` when sampling without replacement from a bank."""
    return finite_pool_bon_plugin_rows(
        torch.ones(scores.shape, dtype=torch.int64), torch.as_tensor(scores),
        torch.as_tensor(values, dtype=torch.float64), GRID).numpy()


def iid_rows(scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Exact per-prompt BoN value when sampling with replacement from a bank."""
    s = torch.as_tensor(scores)
    p = torch.full_like(s, 1 / s.shape[1])
    y = torch.as_tensor(values, dtype=torch.float64)
    return np.stack([(true_bon_policy(p, s, N) * y).sum(1).numpy() for N in GRID], axis=1)


def run(model: str, selector: str, scores_root: Path, outcomes_root: Path, output: Path,
        nuisance: str = SPLINE, nuisance_root: Path | None = None,
        designs: tuple[str, ...] = ("fixed_bank", "iid_empirical")) -> None:
    """Write missing per-seed records under ``output/<model>/<selector>/<nuisance>/``.

    With ``nuisance_root``, stored validation predictions replace the spline
    and only fixed-bank calibration is evaluated (independent reward functions).
    """
    torch.set_num_threads(1)
    model_id = REWARD_MODEL_IDS[selector]
    s = {split: load_table(scores_root / split, "score", model_id)
         for split in ("train", "validation", "test")}
    y = {split: load_table(outcomes_root / split, "correct") for split in s}
    if nuisance_root is None:
        actions = rng(model, "training").integers(0, s["train"].shape[1], len(s["train"]))
        prompts = np.arange(len(actions))
        spline = fit_score_spline(s["train"][prompts, actions], y["train"][prompts, actions])
        mu = spline.predict_proba(s["validation"].reshape(-1, 1))[:, 1].reshape(s["validation"].shape)
    else:
        mu = load_table(nuisance_root / "validation", "score")
        designs = ("fixed_bank",)
    root = output / model / selector / nuisance
    scores, outcomes = s["validation"], y["validation"]
    count, pool = scores.shape
    val, test = exact_rows(scores, outcomes), exact_rows(s["test"], y["test"])
    plugin = exact_rows(scores, mu)
    reward = exact_rows(scores, scores).mean(0)
    population = iid_rows(scores, outcomes)
    oracles = [dict(design=design, split=split, N=N, value=float(rows[:, j].mean()),
                    prompt_se=float(rows[:, j].std(ddof=1) / np.sqrt(len(rows))),
                    expected_score=None if score is None else float(score[j]))
               for design, split, rows, score in [
                   ("fixed_bank", "validation", val, reward),
                   ("fixed_bank", "test", test, exact_rows(s["test"], s["test"]).mean(0)),
                   ("iid_empirical", "validation", population, None)]
               for j, N in enumerate(GRID)]
    root.mkdir(parents=True, exist_ok=True)
    (root / "oracles.json").write_text(json.dumps(oracles, indent=1))
    ratio_table = finite_pool_bon_ratio_table(torch.as_tensor(scores), GRID).numpy()
    test_gain = np.array([(test[:, j] - test[:, 0]).mean() for j in range(len(GRID))])
    val_gain = np.array([(val[:, j] - val[:, 0]).mean() for j in range(len(GRID))])
    for design in designs:
        for seed in range(REPETITIONS[design]):
            path = root / design / f"seed_{seed:04d}.json.gz"
            if path.exists():
                continue
            if design == "fixed_bank":
                a = rng(model, design, seed, "logged").integers(pool, size=count)
                weights = ratio_table[np.arange(count), a]
                logged_y, logged_mu = outcomes[np.arange(count), a], mu[np.arange(count), a]
                dm, truth = plugin, val.mean(0)
            else:
                n = count
                x = rng(model, design, seed, "prompts").integers(count, size=n)
                a = rng(model, design, seed, "logged").integers(pool, size=n)
                auxiliary = rng(model, design, seed, "auxiliary").integers(pool, size=(n, AUXILIARY_DRAWS))
                slots = np.concatenate([a[:, None], auxiliary], axis=1)
                counts = torch.ones((n, AUXILIARY_DRAWS + 1), dtype=torch.int64)
                counts[:, 0] = 0  # slot 0 is the logged response, not part of the pool
                logged_y, logged_mu = outcomes[x, a], mu[x, a]
                rows = finite_pool_scalar_score_components(
                    torch.zeros(n, dtype=torch.int64), torch.as_tensor(logged_y), counts,
                    torch.as_tensor(scores[x[:, None], slots]), torch.as_tensor(mu[x[:, None], slots]),
                    GRID)
                dm, weights = rows.plugin.numpy(), rows.ratio.numpy()
                truth = population.mean(0)
            matrices = dict(dm=dm, ipw=weights * logged_y[:, None],
                            dr=dm + weights * (logged_y - logged_mu)[:, None])
            selection = dict(gain=test_gain, bound_gain=val_gain, expected_score=reward)
            record = evaluate_rows(
                matrices, GRID, truth,
                lambda _, estimator: rng(model, selector, design, seed, estimator, "bootstrap"),
                BOOTSTRAP, ALPHA, **(selection if design == "fixed_bank" and nuisance == SPLINE else {}))
            record["overlap"] = overlap_rows(weights, GRID)
            write_record(path, record)


def aggregate(records: Path, output: Path, independent_output: Path | None = None) -> None:
    """Write the GSM8K tables read by ``plotting/figures.ipynb``."""
    paths = sorted(records.glob("*/*/*/*/seed_*.json.gz"))
    if not paths:
        raise FileNotFoundError(f"no GSM8K records below {records}")
    fields = dict(model=lambda p: p.parts[-5], selector=lambda p: p.parts[-4],
                  nuisance=lambda p: p.parts[-3], design=lambda p: p.parts[-2],
                  seed=lambda p: int(p.name[5:9]))
    calibration = read_records(paths, "calibration", **fields)
    primary = calibration[calibration.nuisance == SPLINE]
    keys = ["model", "selector", "design", "estimator", "N"]
    write_csv(calibration_summary(primary, keys), output / "calibration_summary.csv.gz")
    fixed = [p for p in paths if p.parts[-3] == SPLINE and p.parts[-2] == "fixed_bank"]
    selection = read_records(fixed, "selection", **fields)
    write_csv(selection_summary(selection, ["model", "selector", "estimator", "selection_rule",
                                            "N_max"]), output / "selection_summary_prefixes.csv.gz")
    overlap = read_records([p for p in paths if p.parts[-3] == SPLINE], "overlap", **fields)
    write_csv(overlap[["model", "selector", "design", "seed", "N", "ess", "weight_max",
                       "weight_p99", "nonzero_rate"]], output / "overlap.csv.gz")
    oracles = pd.concat([pd.DataFrame(json.loads(p.read_text())).assign(
        model=p.parts[-4], selector=p.parts[-3]) for p in sorted(records.glob(f"*/*/{SPLINE}/oracles.json"))])
    write_csv(oracles[["model", "selector", "design", "split", "N", "value", "prompt_se",
                       "expected_score"]], output / "oracles.csv.gz")
    independent = calibration[calibration.nuisance != SPLINE]
    if independent_output is not None and len(independent):
        write_csv(calibration_summary(independent, ["model", "selector", "nuisance", "estimator", "N"]),
                  independent_output / "calibration_summary.csv")
