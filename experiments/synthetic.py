"""Controlled rare-tail sweeps: per-seed records and paper aggregates.

Seventeen unique cases vary score error, reward-function error, validation
size ``n`` and auxiliary pool size ``M`` one at a time around the default
(0.65, 0.4, 4096, 256); the ``M`` sweep uses a perfect reward function.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bon.estimators import finite_pool_scalar_score_components
from dgp.synthetic_dgp import make_rare_tail, rare_tail_oracle, sample_rare_tail, true_bon_policy

from .common import (
    calibration_summary, evaluate_rows, read_records, selection_summary, write_csv, write_record,
)

GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SEED = 20260919
REPETITIONS = 300
BOOTSTRAP = 2000
ALPHA = 0.05
DEFAULT = dict(score_distortion=0.65, nuisance_distortion=0.4, n_val=4096, M=256)
SWEEPS = dict(
    score_distortion=(0.0, 0.55, 0.6, 0.65, 0.7, 0.8),
    nuisance_distortion=(-0.2, 0.0, 0.2, 0.4, 0.6),
    n_val=(256, 1024, 4096, 16384, 65536),
    M=(256, 512, 1024, 4096),
)
SWEEP_NAMES = dict(score_distortion="score", nuisance_distortion="nuisance", n_val="n_val", M="M")


def cases() -> list[dict]:
    """Return the unique sweep cases; shared sweep origins are evaluated once."""
    unique: dict[str, dict] = {}
    for field, values in SWEEPS.items():
        for value in values:
            case = dict(DEFAULT, **({"nuisance_distortion": 0.0} if field == "M" else {}))
            case[field] = value
            case_id = "s{score_distortion:g}_d{nuisance_distortion:g}_n{n_val}_m{M}".format(**case)
            unique.setdefault(case_id, dict(case, case_id=case_id, sweeps=[]))
            unique[case_id]["sweeps"].append(SWEEP_NAMES[field])
    return list(unique.values())


def run_seed(case: dict, seed: int) -> dict:
    """Evaluate Plug-in, BoN-IPW and BoN-DR with both selection rules for one seed."""
    world = make_rare_tail(case["score_distortion"], case["nuisance_distortion"])
    oracle = {key: value.numpy() for key, value in rare_tail_oracle(world, GRID).items()}
    replication = SEED + seed
    x, a, y, counts = sample_rare_tail(world, case["n_val"], case["M"], replication)
    rows = finite_pool_scalar_score_components(
        a, y, counts, world.scores[x], world.nuisance[x], GRID)
    matrices = dict(dm=rows.plugin.numpy(), ipw=(rows.ratio * y[:, None]).numpy(),
                    dr=rows.dr.numpy())
    gain = oracle["truth"] - oracle["truth"][0]
    return evaluate_rows(
        matrices, GRID, oracle["truth"],
        lambda index, _: np.random.default_rng(np.random.SeedSequence([replication, index, 731])),
        BOOTSTRAP, ALPHA, gain=gain, bound_gain=gain, expected_score=oracle["expected_score"])


def run(output: Path, case_ids: list[str] | None, seeds: range) -> None:
    """Write missing per-seed records under ``output/<case_id>/``."""
    torch.set_num_threads(1)
    for case in cases():
        if case_ids and case["case_id"] not in case_ids:
            continue
        for seed in seeds:
            path = output / case["case_id"] / f"seed_{seed:04d}.json.gz"
            if not path.exists():
                write_record(path, run_seed(case, seed))


def efficiency_comparison(calibration: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Compare finite-M BoN-DR variance with the infinite-M oracle-EIF estimator.

    The oracle estimator reuses the same logged draws with exact ``pi_N/pi_b``
    and exact outcome means; seed-bootstrap intervals give Monte Carlo error.
    """
    world = make_rare_tail(DEFAULT["score_distortion"], 0.0)
    policies = torch.stack([true_bon_policy(world.probabilities, world.scores, N) for N in GRID], 2)
    values = (policies * world.quality[:, :, None]).sum(1)
    ratios = policies / world.probabilities[:, :, None]
    reference = []
    for seed in range(REPETITIONS):
        # M only affects the auxiliary stream, which the oracle estimator ignores.
        x, a, y, _ = sample_rare_tail(world, DEFAULT["n_val"], 1, SEED + seed)
        means = (values[x] + ratios[x, a] * (y - world.quality[x, a])[:, None]).mean(0).numpy()
        reference.extend(dict(seed=seed, N=N, estimate=means[j]) for j, N in enumerate(GRID))
    reference = pd.DataFrame(reference)
    m_cases = table[table.sweeps.str.split("|").apply(lambda v: "M" in v)]
    finite = calibration[calibration.estimator == "dr"].merge(m_cases[["case_id", "M"]])
    paired = finite[["case_id", "M", "seed", "N", "estimate"]].merge(
        reference, on=["seed", "N"], suffixes=("_finite", "_oracle"))
    rng = np.random.default_rng(SEED)
    rows = []
    for (case_id, M, N), group in paired.groupby(["case_id", "M", "N"]):
        f, o = group.estimate_finite.to_numpy(), group.estimate_oracle.to_numpy()
        draws = rng.integers(len(group), size=(2000, len(group)))
        samples = f[draws].var(1, ddof=1) / o[draws].var(1, ddof=1)
        rows.append(dict(
            case_id=case_id, M=M, N=N, repetitions=len(group),
            oracle_empirical_variance=o.var(ddof=1), finite_empirical_variance=f.var(ddof=1),
            paired_variance_ratio=f.var(ddof=1) / o.var(ddof=1),
            paired_ratio_mc_lower=float(np.quantile(samples, 0.025)),
            paired_ratio_mc_upper=float(np.quantile(samples, 0.975)),
            paired_difference_mean=float((f - o).mean()),
            paired_difference_mcse=float((f - o).std(ddof=1) / np.sqrt(len(group)))))
    return pd.DataFrame(rows)


def aggregate(records: Path, output: Path) -> None:
    """Write the synthetic tables read by ``plotting/figures.ipynb``."""
    table = pd.DataFrame(cases()).sort_values("case_id", ignore_index=True)
    paths = sorted(records.glob("*/seed_*.json.gz"))
    found = {(p.parent.name, int(p.name[5:9])) for p in paths}
    missing = {(c, s) for c in table.case_id for s in range(REPETITIONS)} - found
    if missing:
        raise ValueError(f"{len(missing)} synthetic records are missing, e.g. {min(missing)}")
    fields = dict(case_id=lambda p: p.parent.name, seed=lambda p: int(p.name[5:9]))
    calibration = read_records(paths, "calibration", **fields)
    selection = read_records(paths, "selection", **fields)
    table["sweeps"] = table.sweeps.str.join("|")
    oracles = pd.concat(
        [pd.DataFrame(dict(case_id=c.case_id, N=GRID, **{
            k: v.numpy() for k, v in rare_tail_oracle(
                make_rare_tail(c.score_distortion, c.nuisance_distortion), GRID).items()}))
         for c in table.itertuples()], ignore_index=True)
    summary = calibration_summary(calibration, ["case_id", "estimator", "N"]).merge(
        table[["case_id", "n_val"]]).merge(oracles[["case_id", "N", "efficient_variance"]])
    summary["variance_over_efficiency_bound"] = (
        summary.n_val * summary.empirical_variance / summary.efficient_variance)
    write_csv(table, output / "cases.csv.gz")
    write_csv(oracles, output / "oracles.csv.gz")
    write_csv(summary, output / "calibration_summary.csv.gz")
    write_csv(selection_summary(selection, ["case_id", "estimator", "selection_rule", "N_max"],
                                regret=True), output / "selection_summary.csv.gz")
    write_csv(efficiency_comparison(calibration, table), output / "efficiency_comparison.csv.gz")
