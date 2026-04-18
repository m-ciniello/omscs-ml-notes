"""Cross-experiment aggregation.

`aggregate.py` handles within-experiment stats (mean/IQR/CI over seeds).
This module handles *between*-experiment stats — specifically, loading a
full HP sweep (all variants sharing a name prefix or tag) and producing a
tidy structure that maps each sweep value to its seed-aggregated metrics.

Typical usage:

    from src.configs import list_experiments
    from src.experiments.compare import load_sweep, sweep_to_dataframe

    names = list_experiments(name_prefix="blackjack_vi_gamma_sweep")
    sweep = load_sweep(names, sweep_path="gamma")
    df = sweep_to_dataframe(sweep)
    # df columns: sweep_value, eval_return_mean, eval_return_ci95_low, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.experiments.aggregate import (
    AggregatedMetric,
    SeedRun,
    aggregate_scalar,
    load_config,
    load_runs,
)
from src.experiments.runner import DEFAULT_RESULTS_ROOT


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    """One (sweep_value, seed-aggregated metrics) pair.

    Carries the underlying per-seed runs so downstream code can also extract
    curve-shaped metrics (learning curves) if it wants to.
    """
    experiment_name: str
    sweep_value: Any
    config: dict
    runs: list[SeedRun]


@dataclass
class Sweep:
    """A collection of SweepPoints, sorted by sweep value when numeric."""
    sweep_path: str
    points: list[SweepPoint]

    def __iter__(self):
        return iter(self.points)

    def __len__(self):
        return len(self.points)

    @property
    def values(self) -> list:
        return [p.sweep_value for p in self.points]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sweep(
    experiment_names: list[str],
    sweep_path: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Sweep:
    """Load all runs for each experiment and extract the swept HP value.

    Args:
        experiment_names: names of the sweep variants. Typically produced by
            `configs.list_experiments(name_prefix=...)`.
        sweep_path: dotted path used to extract the HP value from each
            experiment's snapshotted config. Must match the path passed to
            `register_sweep` at registration time.
        results_root: where result directories live.

    Returns:
        A Sweep with one SweepPoint per experiment, sorted ascending by
        sweep_value when values are numerically comparable.
    """
    points: list[SweepPoint] = []
    for name in experiment_names:
        config = load_config(name, results_root=results_root)
        value = _read_path(config, sweep_path)
        runs = load_runs(name, results_root=results_root)
        points.append(SweepPoint(
            experiment_name=name,
            sweep_value=value,
            config=config,
            runs=runs,
        ))

    try:
        points.sort(key=lambda p: float(p.sweep_value))
    except (TypeError, ValueError):
        points.sort(key=lambda p: str(p.sweep_value))

    return Sweep(sweep_path=sweep_path, points=points)


def _read_path(obj: dict, path: str) -> Any:
    """Walk a dotted path into a nested dict (from asdict(ExperimentSpec))."""
    cur: Any = obj
    for part in path.split("."):
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------------
# Per-sweep aggregation
# ---------------------------------------------------------------------------

def aggregate_sweep_metric(
    sweep: Sweep,
    extractor: Callable[[SeedRun], float],
    name: str = "metric",
) -> dict[Any, AggregatedMetric]:
    """Compute a per-sweep-value aggregated metric.

    Args:
        sweep: loaded Sweep.
        extractor: callable (SeedRun) -> scalar, aggregated across seeds for
            each sweep point.
        name: label carried through into AggregatedMetric.

    Returns:
        Dict mapping sweep_value -> AggregatedMetric.
    """
    return {
        p.sweep_value: aggregate_scalar(p.runs, extractor, name=name)
        for p in sweep.points
    }


# ---------------------------------------------------------------------------
# Convenience: pretty-printing and DataFrame export
# ---------------------------------------------------------------------------

# Standard scalars we extract for most sweeps. Extractor returns NaN if the
# summary lacks the key — this happens for DP agents which have no training
# curves, and is fine.
_STANDARD_EXTRACTORS: dict[str, Callable[[SeedRun], float]] = {
    "eval_return_mean": lambda r: r.summary.get("eval_return_mean", np.nan),
    "eval_return_std": lambda r: r.summary.get("eval_return_std", np.nan),
    "train_return_mean": lambda r: r.summary.get("train_return_mean", np.nan),
    "wall_clock_seconds": lambda r: r.summary.get("wall_clock_seconds", np.nan),
}


def sweep_to_dataframe(sweep: Sweep):
    """Turn a Sweep into a tidy pandas DataFrame for plotting / tables.

    One row per (sweep_value, seed) pair for long-form metrics, plus
    point-aggregated columns (mean/CI) for short-form. Imports pandas
    lazily so the rest of the package works without it.
    """
    import pandas as pd

    rows = []
    for point in sweep.points:
        for run in point.runs:
            row = {
                "experiment": point.experiment_name,
                "sweep_value": point.sweep_value,
                "seed": run.seed,
            }
            row.update({k: fn(run) for k, fn in _STANDARD_EXTRACTORS.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def print_sweep_summary(sweep: Sweep) -> None:
    """Print a human-readable per-value summary of standard metrics."""
    print(f"\n=== Sweep over {sweep.sweep_path} "
          f"({len(sweep)} values) ===")
    header = f"{'value':>10s}  {'eval_mean':>10s}  {'eval_std':>10s}  " \
             f"{'ci95_lo':>10s}  {'ci95_hi':>10s}  {'wall':>8s}"
    print(header)
    print("-" * len(header))
    eval_aggs = aggregate_sweep_metric(
        sweep, _STANDARD_EXTRACTORS["eval_return_mean"], "eval_return_mean"
    )
    wall_aggs = aggregate_sweep_metric(
        sweep, _STANDARD_EXTRACTORS["wall_clock_seconds"], "wall_clock_seconds"
    )
    for point in sweep.points:
        v = point.sweep_value
        e = eval_aggs[v]
        w = wall_aggs[v]
        print(f"{_fmt(v):>10s}  {e.mean:>10.4f}  {e.std:>10.4f}  "
              f"{e.ci95_low:>10.4f}  {e.ci95_high:>10.4f}  {w.mean:>8.3f}")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)
