"""Model-neutral loading for stored reward-score artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch


SCORE_COLUMNS = ("question_id", "sample_id", "score")


def load_score_artifacts(
    root: Path, *, expected_model_id: str | None = None
) -> pd.DataFrame:
    """Load score Parquets and optionally validate their reward-model identity."""
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and expected_model_id is not None:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("model_id") not in {None, expected_model_id}:
            raise ValueError("score manifest uses a different reward model")
    paths = sorted((root / "rewards").glob("q*.parquet"))
    if not paths:
        paths = sorted((root / "shards").glob("q*/s*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no score Parquet files below {root}")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    missing = set(SCORE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"score columns are missing {sorted(missing)}")
    frame = frame.loc[:, SCORE_COLUMNS].sort_values(["question_id", "sample_id"])
    if frame.duplicated(["question_id", "sample_id"]).any() or not torch.isfinite(
        torch.as_tensor(frame.score.to_numpy())
    ).all():
        raise ValueError("scores contain duplicate keys or non-finite values")
    return frame.reset_index(drop=True)
