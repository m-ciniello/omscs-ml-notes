"""Multi-seed experiment runner and result loaders.

Writes each spec's per-seed results to disk, and exposes small helpers for
reading them back from figure/analysis scripts.

Per-experiment on-disk layout:

    results/<name>/
        config.json
        seed_<i>/
            result.pkl       # full RunResult dict (source of truth)
            summary.json     # human-readable scalar sidecar

Sweep variants live under a shared parent (see `ExperimentSpec.results_path_parts`).
"""

from __future__ import annotations

import json
import pickle
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from src.agents import build_agent
from src.configs import ExperimentSpec, get
from src.envs import build_env


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def experiment_dir(
    name_or_spec: str | ExperimentSpec,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Resolve the on-disk results directory for an experiment."""
    spec = get(name_or_spec) if isinstance(name_or_spec, str) else name_or_spec
    parts = spec.results_path_parts or (spec.name,)
    return Path(results_root, *parts)


def run_spec(
    spec: ExperimentSpec,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Run every seed for a spec. Returns the experiment directory.

    If any seed already has a `result.pkl` on disk we raise `FileExistsError`
    rather than silently skipping — the caller is expected to `rm -rf` the
    experiment (or pass `overwrite=True`) before rerunning. This avoids ever
    serving stale numbers from a previous code/config state.
    """
    exp_dir = experiment_dir(spec, results_root=results_root)

    # Pre-flight: refuse to touch anything on disk (including config.json)
    # until we know every seed slot is free. Avoids leaving a mismatched
    # snapshot next to stale per-seed pickles when the user forgot to rm.
    if not overwrite:
        for seed in spec.seeds:
            result_path = exp_dir / f"seed_{seed}" / "result.pkl"
            if result_path.exists():
                raise FileExistsError(
                    f"{result_path} already exists. Delete the experiment "
                    f"directory (rm -rf {exp_dir}) or pass --overwrite to "
                    f"rerun."
                )

    exp_dir.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(exp_dir, spec)

    if verbose:
        print(f"[run] {spec.name}  (seeds={list(spec.seeds)}, "
              f"env={spec.env.name}, agent={spec.agent.name})")

    for seed in spec.seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _run_single_seed(spec, seed, seed_dir, verbose=verbose)

    return exp_dir


def _run_single_seed(
    spec: ExperimentSpec,
    seed: int,
    seed_dir: Path,
    *,
    verbose: bool = True,
) -> None:
    _seed_everything(seed)

    env = build_env(spec.env, seed=seed)
    agent = build_agent(spec.agent)

    t0 = time.perf_counter()
    result = agent.run(
        env,
        n_episodes=spec.n_episodes,
        eval_episodes=spec.eval_episodes,
        gamma=spec.gamma,
        seed=seed,
    )
    wall = time.perf_counter() - t0
    result.setdefault("wall_clock_seconds", wall)

    with open(seed_dir / "result.pkl", "wb") as f:
        pickle.dump(result, f)
    with open(seed_dir / "summary.json", "w") as f:
        json.dump(_summarise(result), f, indent=2)

    if verbose:
        summary = _summarise(result)
        mean_eval = summary.get("eval_return_mean", float("nan"))
        print(f"  seed {seed}: eval_return_mean={mean_eval:.3f}  "
              f"wall={result['wall_clock_seconds']:.2f}s")


def _summarise(result: dict) -> dict[str, Any]:
    """JSON-able scalar metrics from a RunResult dict."""
    summary: dict[str, Any] = {
        "wall_clock_seconds": float(result.get("wall_clock_seconds", 0.0)),
    }
    for prefix in ("train", "eval"):
        returns = result.get(f"{prefix}_returns", [])
        steps = result.get(f"{prefix}_steps", [])
        if returns:
            arr = np.asarray(returns, dtype=float)
            summary[f"{prefix}_return_mean"] = float(arr.mean())
            summary[f"{prefix}_return_std"] = float(arr.std())
            summary[f"{prefix}_n_episodes"] = int(len(arr))
        if steps:
            arr = np.asarray(steps, dtype=float)
            summary[f"{prefix}_steps_mean"] = float(arr.mean())
            summary[f"{prefix}_steps_std"] = float(arr.std())
    return summary


def _seed_everything(seed: int) -> None:
    """Seed random, numpy, and (lazily) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _write_config_snapshot(exp_dir: Path, spec: ExperimentSpec) -> None:
    with open(exp_dir / "config.json", "w") as f:
        json.dump(asdict(spec), f, indent=2, default=str)


class SeedRun(NamedTuple):
    """One seed's payload: (seed, full RunResult dict, summary.json dict)."""
    seed: int
    result: dict
    summary: dict


def load_runs(
    name_or_spec: str | ExperimentSpec,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> list[SeedRun]:
    """Load every seed's result for an experiment (sorted by seed).

    Stat aggregation (mean / CI / IQR) is done at the call site because the
    exact shape differs per figure; a generic helper didn't pull its weight.
    """
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
