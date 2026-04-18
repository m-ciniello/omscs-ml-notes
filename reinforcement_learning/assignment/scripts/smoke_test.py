"""Phase 0 smoke test: verify the harness works end-to-end.

Runs the `smoke_gridworld_random` experiment across 3 seeds, aggregates
the results, and prints a summary. No ML happens here — this is purely a
test that:

1. Configs register correctly and can be looked up by name.
2. The runner builds env + agent from specs, runs the agent, and writes
   `result.pkl` + `summary.json` for each seed.
3. The aggregator loads those files and computes seed-aggregated stats.
4. The on-disk layout matches what `runner.py` documents.

Usage:

    python -m scripts.smoke_test

Expects to be run from the `assignment/` directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src` is importable when this script is invoked directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.aggregate import (  # noqa: E402
    aggregate_curve,
    load_runs,
    print_summary,
)
from src.experiments.runner import run_experiment  # noqa: E402


EXPERIMENT_NAME = "smoke_gridworld_random"


def main() -> int:
    print("Phase 0 smoke test — runner + aggregator on gridworld + random agent")
    print("-" * 72)

    exp_dir = run_experiment(EXPERIMENT_NAME, overwrite=True)
    print(f"\nResults dumped to: {exp_dir.relative_to(REPO_ROOT)}")

    print_summary(EXPERIMENT_NAME)

    runs = load_runs(EXPERIMENT_NAME)
    train_curve = aggregate_curve(runs, key="train_returns",
                                  name="training return")
    print(f"\nLearning curve shape: {train_curve.matrix.shape}  "
          f"(n_seeds × n_episodes)")
    print(f"  mean return @ ep 0 : {train_curve.mean[0]:.3f}  "
          f"(IQR {train_curve.iqr_low[0]:.3f}, {train_curve.iqr_high[0]:.3f})")
    print(f"  mean return @ ep-1 : {train_curve.mean[-1]:.3f}  "
          f"(IQR {train_curve.iqr_low[-1]:.3f}, {train_curve.iqr_high[-1]:.3f})")

    checks = _validate_on_disk_layout(exp_dir, n_seeds=len(runs))
    print(f"\nOn-disk layout checks:")
    for name, ok in checks.items():
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {name}")

    if not all(checks.values()):
        return 1
    print("\nSmoke test passed.")
    return 0


def _validate_on_disk_layout(exp_dir: Path, n_seeds: int) -> dict[str, bool]:
    """Check that the runner wrote the files we expect, exactly where we expect."""
    checks = {
        "config.json present": (exp_dir / "config.json").is_file(),
        f"seed directories ({n_seeds} expected)":
            sum(1 for _ in exp_dir.glob("seed_*")) == n_seeds,
    }
    for seed_dir in exp_dir.glob("seed_*"):
        seed = seed_dir.name
        checks[f"{seed}/result.pkl"] = (seed_dir / "result.pkl").is_file()
        checks[f"{seed}/summary.json"] = (seed_dir / "summary.json").is_file()
    return checks


if __name__ == "__main__":
    sys.exit(main())
