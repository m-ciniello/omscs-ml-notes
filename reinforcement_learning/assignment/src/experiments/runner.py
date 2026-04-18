"""Multi-seed experiment runner.

One function does one thing: given an experiment name, run it across all
seeds and dump the results to disk. No aggregation, no plotting — just
"turn the crank and write files."

On-disk layout for a standalone experiment `foo`:

```
results/
  foo/
    config.json           # snapshot of ExperimentSpec.to_dict()
    git_sha.txt           # commit SHA at run time (if available)
    seed_0/
      result.pkl          # full RunResult (policy, Q, history, etc.)
      summary.json        # human-readable scalar metrics
    seed_1/
      ...
```

On-disk layout for a sweep `bar_sweep` over values [v1, v2, ...]:

```
results/
  bar_sweep/
    sweep_manifest.json   # describes the sweep (path, variants, values)
    v1/
      config.json
      git_sha.txt
      seed_0/ ...
    v2/
      ...
```

The path is determined by `ExperimentSpec.results_path_parts`. If empty
(the default), we fall back to `(spec.name,)` so manually-registered
single-point specs don't need edits.

The `result.pkl` is the source of truth. `summary.json` is a convenience
sidecar for grepping / quick inspection.
"""

from __future__ import annotations

import json
import pickle
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.agents import build_agent
from src.configs import EXPERIMENTS, ExperimentSpec, get
from src.envs import build_env


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results"


def experiment_dir(
    name_or_spec: str | ExperimentSpec,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Resolve the results directory for an experiment.

    Uses `spec.results_path_parts` when set (sweep variants live under a
    shared parent), otherwise falls back to `(spec.name,)`. This is the
    single place that logic lives — everywhere else that needs to find an
    experiment's results on disk should call this helper.
    """
    spec = get(name_or_spec) if isinstance(name_or_spec, str) else name_or_spec
    parts = spec.results_path_parts or (spec.name,)
    return Path(results_root, *parts)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_experiment(
    name: str,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    """Run every seed for the named experiment; return the experiment directory.

    Args:
        name: key into the EXPERIMENTS registry in `src/configs.py`.
        results_root: directory to dump per-seed results. Default `results/`.
        overwrite: if False, seeds whose `result.pkl` already exists are skipped.
        verbose: print a one-line status update per seed.

    Returns:
        The path to the experiment's results directory.
    """
    spec = get(name)
    exp_dir = experiment_dir(spec, results_root=results_root)
    exp_dir.mkdir(parents=True, exist_ok=True)

    _write_config_snapshot(exp_dir, spec)
    _write_git_sha(exp_dir)
    _maybe_write_sweep_manifest(spec, results_root=results_root)

    if verbose:
        print(f"[run] {spec.name}  (seeds={list(spec.seeds)}, "
              f"env={spec.env.name}, agent={spec.agent.name})")

    for seed in spec.seeds:
        seed_dir = exp_dir / f"seed_{seed}"
        result_path = seed_dir / "result.pkl"
        if result_path.exists() and not overwrite:
            if verbose:
                print(f"  seed {seed}: cached, skipping "
                      f"(use overwrite=True to rerun)")
            continue
        seed_dir.mkdir(exist_ok=True)
        _run_single_seed(spec, seed, seed_dir, verbose=verbose)

    return exp_dir


def run_experiments(
    names: list[str],
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """Run multiple experiments sequentially. Simple convenience wrapper."""
    return [
        run_experiment(n, results_root=results_root, overwrite=overwrite,
                       verbose=verbose)
        for n in names
    ]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _run_single_seed(
    spec: ExperimentSpec,
    seed: int,
    seed_dir: Path,
    *,
    verbose: bool = True,
) -> None:
    """Run one (experiment, seed) pair and write result.pkl + summary.json."""
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

    # Agents typically report their own wall clock; fall back to the outer
    # measurement if they don't.
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
    """Extract JSON-able scalar metrics from a RunResult.

    Keeps a human-readable sidecar next to the pickle so the directory tree
    can be skimmed without writing any Python.
    """
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
    """Seed every randomness source we might use.

    Torch is imported lazily so we don't pay the import cost for non-DQN
    experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no CUDA in sandbox
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _write_config_snapshot(exp_dir: Path, spec: ExperimentSpec) -> None:
    """Save the experiment spec verbatim so later runs can diff against it."""
    with open(exp_dir / "config.json", "w") as f:
        json.dump(asdict(spec), f, indent=2, default=str)


def _write_git_sha(exp_dir: Path) -> None:
    """Record the current git SHA for reproducibility (best-effort)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        with open(exp_dir / "git_sha.txt", "w") as f:
            f.write(sha + "\n")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


def _maybe_write_sweep_manifest(
    spec: ExperimentSpec,
    *,
    results_root: Path,
) -> None:
    """If spec is a sweep variant, write a manifest at the sweep parent dir.

    A "sweep variant" is identified by having `len(results_path_parts) >= 2`:
    the first component is the sweep's shared parent directory. The manifest
    enumerates every registered variant that shares that parent and lists
    the sweep paths they vary, so the sweep-level folder is self-describing.

    Idempotent: rewrites the manifest every call, which is cheap and keeps
    it current as new variants are registered.
    """
    parts = spec.results_path_parts
    if len(parts) < 2:
        return
    sweep_parent_name = parts[0]
    parent_dir = results_root / sweep_parent_name

    variants: list[dict[str, Any]] = []
    sweep_paths: set[str] = set()
    for other in EXPERIMENTS.values():
        other_parts = other.results_path_parts
        if len(other_parts) >= 2 and other_parts[0] == sweep_parent_name:
            variants.append({
                "name": other.name,
                "value_fragment": other_parts[1],
            })
            for tag in other.tags:
                if tag.startswith("sweep:"):
                    sweep_paths.add(tag.split("sweep:", 1)[1])

    manifest = {
        "sweep_name": sweep_parent_name,
        "sweep_paths": sorted(sweep_paths),
        "variants": sorted(variants, key=lambda v: v["name"]),
        "description": spec.description,
    }
    parent_dir.mkdir(parents=True, exist_ok=True)
    with open(parent_dir / "sweep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
