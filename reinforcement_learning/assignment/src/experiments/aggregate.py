"""Multi-seed result aggregation.

Loads per-seed result pickles from `results/<experiment>/` and produces
seed-aggregated metrics: means, medians, IQR bands, and 95% confidence
intervals. Used by figure-generation scripts and the report writeup.

The FAQ requires mean + variability bands (IQR or 95% CI) across ≥ 5 seeds
for every reported number. This module is the single place that logic lives.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.runner import DEFAULT_RESULTS_ROOT, experiment_dir


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class SeedRun:
    """One seed's complete result."""
    seed: int
    result: dict  # the full RunResult pickle
    summary: dict  # the summary.json sidecar


@dataclass
class AggregatedMetric:
    """A metric aggregated over seeds (seed-scalar, e.g. final eval return).

    For curve-shaped metrics (per-episode returns) use `AggregatedCurve`.
    """
    name: str
    values: np.ndarray  # shape (n_seeds,)
    mean: float
    std: float
    median: float
    iqr_low: float   # 25th percentile
    iqr_high: float  # 75th percentile
    ci95_low: float  # mean - 1.96 * sem
    ci95_high: float


@dataclass
class AggregatedCurve:
    """A learning curve aggregated over seeds.

    Curves are assumed to be aligned by episode index; shorter runs are
    right-padded with NaN so per-episode statistics degrade gracefully if
    seeds had different lengths.
    """
    name: str
    matrix: np.ndarray    # shape (n_seeds, n_episodes) with NaN padding
    mean: np.ndarray      # shape (n_episodes,)
    median: np.ndarray
    iqr_low: np.ndarray   # 25th percentile at each episode
    iqr_high: np.ndarray  # 75th percentile at each episode
    ci95_low: np.ndarray
    ci95_high: np.ndarray


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(
    experiment_name: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> list[SeedRun]:
    """Load every seed's result for an experiment.

    Returns a list of SeedRun, sorted by seed number. Errors if no
    results are found.
    """
    exp_dir = experiment_dir(experiment_name, results_root=results_root)
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


def load_config(
    experiment_name: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> dict[str, Any]:
    """Load the snapshotted config for an experiment."""
    exp_dir = experiment_dir(experiment_name, results_root=results_root)
    with open(exp_dir / "config.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_scalar(
    runs: list[SeedRun],
    extractor,
    name: str = "metric",
) -> AggregatedMetric:
    """Aggregate a per-seed scalar.

    Args:
        runs: list of SeedRun loaded via `load_runs`.
        extractor: callable taking a SeedRun and returning a scalar.
        name: label carried through in the output.
    """
    values = np.array([float(extractor(r)) for r in runs])
    return _scalar_summary(values, name)


def aggregate_curve(
    runs: list[SeedRun],
    key: str,
    name: str | None = None,
) -> AggregatedCurve:
    """Aggregate a per-seed sequence (e.g. `train_returns`) into a curve.

    Sequences of different lengths are right-padded with NaN; statistics
    use `nanmean` / `nanpercentile` so padded entries don't bias results.
    """
    sequences = [np.asarray(r.result.get(key, []), dtype=float) for r in runs]
    if not sequences or all(len(s) == 0 for s in sequences):
        raise ValueError(f"No seed provided key {key!r}")

    max_len = max(len(s) for s in sequences)
    matrix = np.full((len(sequences), max_len), np.nan)
    for i, s in enumerate(sequences):
        matrix[i, :len(s)] = s

    mean = np.nanmean(matrix, axis=0)
    median = np.nanmedian(matrix, axis=0)
    iqr_low = np.nanpercentile(matrix, 25, axis=0)
    iqr_high = np.nanpercentile(matrix, 75, axis=0)

    # 95% CI uses SEM = std / sqrt(n_non_nan); falls back to NaN at positions
    # where only one seed contributed (SEM undefined).
    n = np.sum(~np.isnan(matrix), axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = std / np.sqrt(n)
    ci95_low = mean - 1.96 * sem
    ci95_high = mean + 1.96 * sem

    return AggregatedCurve(
        name=name or key,
        matrix=matrix,
        mean=mean,
        median=median,
        iqr_low=iqr_low,
        iqr_high=iqr_high,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
    )


def _scalar_summary(values: np.ndarray, name: str) -> AggregatedMetric:
    """Compute the standard summary stats for a 1-D array."""
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    sem = std / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return AggregatedMetric(
        name=name,
        values=values,
        mean=mean,
        std=std,
        median=float(np.median(values)),
        iqr_low=float(np.percentile(values, 25)),
        iqr_high=float(np.percentile(values, 75)),
        ci95_low=mean - 1.96 * sem,
        ci95_high=mean + 1.96 * sem,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def print_summary(experiment_name: str,
                  results_root: Path = DEFAULT_RESULTS_ROOT) -> None:
    """Dump a short human-readable summary to stdout. Useful for smoke tests."""
    runs = load_runs(experiment_name, results_root=results_root)
    cfg = load_config(experiment_name, results_root=results_root)
    print(f"\n=== {experiment_name} ===")
    print(f"  env: {cfg['env']['name']}   agent: {cfg['agent']['name']}")
    print(f"  seeds: {[r.seed for r in runs]}")
    eval_ret = aggregate_scalar(
        runs,
        extractor=lambda r: r.summary.get("eval_return_mean", float("nan")),
        name="eval_return_mean",
    )
    print(f"  eval return (mean of seed-means): "
          f"{eval_ret.mean:.3f} ± {eval_ret.std:.3f}  "
          f"[95% CI: {eval_ret.ci95_low:.3f}, {eval_ret.ci95_high:.3f}]")
    wall = aggregate_scalar(
        runs,
        extractor=lambda r: r.summary.get("wall_clock_seconds", 0.0),
        name="wall_clock_seconds",
    )
    print(f"  wall clock per seed: {wall.mean:.2f}s "
          f"(median {wall.median:.2f}s)")
