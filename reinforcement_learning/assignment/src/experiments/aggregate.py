"""Load per-seed results and config snapshots from disk.

Every reported figure / table reads one of these two helpers. Stat
aggregation (mean / CI / IQR) is done at the call site because the exact
shape differs per figure; a generic helper didn't pull its weight.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, NamedTuple

from src.experiments.runner import DEFAULT_RESULTS_ROOT, experiment_dir


class SeedRun(NamedTuple):
    """One seed's payload: (seed, full RunResult dict, summary.json dict)."""
    seed: int
    result: dict
    summary: dict


def load_runs(name_or_spec, results_root: Path = DEFAULT_RESULTS_ROOT) -> list[SeedRun]:
    """Load every seed's result for an experiment (sorted by seed)."""
    exp_dir = experiment_dir(name_or_spec, results_root=results_root)
    if not exp_dir.is_dir():
        raise FileNotFoundError(
            f"No results directory at {exp_dir}. Run the experiment first."
        )
    runs: list[SeedRun] = []
    for seed_dir in sorted(exp_dir.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        with open(seed_dir / "result.pkl", "rb") as f:
            result = pickle.load(f)
        with open(seed_dir / "summary.json") as f:
            summary = json.load(f)
        runs.append(SeedRun(seed=seed, result=result, summary=summary))
    if not runs:
        raise FileNotFoundError(
            f"Results directory {exp_dir} exists but contains no seed runs."
        )
    return runs


def load_config(name_or_spec, results_root: Path = DEFAULT_RESULTS_ROOT) -> dict[str, Any]:
    """Load the snapshotted ExperimentSpec (config.json)."""
    exp_dir = experiment_dir(name_or_spec, results_root=results_root)
    with open(exp_dir / "config.json") as f:
        return json.load(f)
