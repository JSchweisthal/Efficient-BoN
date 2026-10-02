"""Entry point for the synthetic and GSM8K paper experiments.

``synthetic`` and ``llm`` write resumable per-seed records; ``aggregate``
turns completed records into the tables under ``plotting/data/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments import llm, synthetic


def seed_range(text: str) -> range:
    """Parse ``START:STOP`` (half-open) or a single seed."""
    start, _, stop = text.partition(":")
    return range(int(start), int(stop) if stop else int(start) + 1)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    syn = commands.add_parser("synthetic", help="rare-tail sweeps (CPU)")
    syn.add_argument("--cases", nargs="+", help="case ids (default: all 17)")
    syn.add_argument("--seeds", type=seed_range, default=range(synthetic.REPETITIONS))
    syn.add_argument("--output", type=Path, default=Path("results/synthetic"))

    real = commands.add_parser("llm", help="GSM8K from stored scores and outcomes (CPU)")
    real.add_argument("--model", required=True, choices=["gemma", "llama", "qwen"])
    real.add_argument("--selector", required=True, choices=sorted(llm.REWARD_MODEL_IDS))
    real.add_argument("--scores", type=Path, required=True,
                      help="directory with train/, validation/, test/ score stores")
    real.add_argument("--outcomes", type=Path, required=True,
                      help="directory with train/, validation/, test/ outcome stores")
    real.add_argument("--nuisance-name", help="label of an independently trained reward function")
    real.add_argument("--nuisance", type=Path,
                      help="directory with a validation/ store of its predictions")
    real.add_argument("--output", type=Path, default=Path("results/llm"))

    agg = commands.add_parser("aggregate", help="rebuild plotting inputs from records")
    agg.add_argument("setting", choices=["synthetic", "llm"])
    agg.add_argument("--records", type=Path)
    agg.add_argument("--output", type=Path, default=Path("plotting/data"))
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "synthetic":
        synthetic.run(args.output, args.cases, args.seeds)
    elif args.command == "llm":
        if (args.nuisance is None) != (args.nuisance_name is None):
            raise SystemExit("--nuisance and --nuisance-name must be given together")
        extra = {} if args.nuisance is None else dict(
            nuisance=args.nuisance_name, nuisance_root=args.nuisance)
        llm.run(args.model, args.selector, args.scores, args.outcomes, args.output, **extra)
    elif args.setting == "synthetic":
        synthetic.aggregate(args.records or Path("results/synthetic"), args.output / "synthetic")
    else:
        llm.aggregate(args.records or Path("results/llm"), args.output / "real",
                      args.output / "independent_nuisance")


if __name__ == "__main__":
    main()
