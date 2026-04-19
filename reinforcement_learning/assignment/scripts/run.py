"""Run every experiment matching a name prefix, then print a results table.

    python scripts/run.py --prefix blackjack_vi_gamma_sweep
    python scripts/run.py --prefix blackjack_vi --no-run

Deterministic outputs, per-experiment eval stats printed at the end. To focus
on a single sweep, just tighten the prefix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.configs import get, list_experiments  # noqa: E402
from src.experiments.runner import load_runs, run_spec  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", type=str, default=None,
                   help="Name prefix (e.g. blackjack_vi_gamma_sweep).")
    p.add_argument("--no-run", action="store_true",
                   help="Skip running; only summarise existing results.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if result.pkl already exists.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    names = list_experiments(name_prefix=args.prefix)
    if not names:
        print("No experiments match the given prefix. Registered:")
        for n in list_experiments():
            print(f"  {n}")
        return 1

    print(f"Matched {len(names)} experiments:")
    for n in names:
        print(f"  {n}")

    if not args.no_run:
        print("\n--- Running ---")
        for name in names:
            run_spec(get(name), overwrite=args.overwrite)

    print("\n--- Results ---")
    _print_results_summary(names)

    return 0


def _collect_stats(name: str) -> dict | None:
    """Mean/std/CI95/wall across seeds for one experiment, or None if not on disk."""
    try:
        runs = load_runs(name)
    except FileNotFoundError:
        return None
    vals = np.array([r.summary.get("eval_return_mean", np.nan)
                     for r in runs], dtype=float)
    walls = np.array([r.summary.get("wall_clock_seconds", 0.0)
                      for r in runs], dtype=float)
    n = len(vals)
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    mean = float(vals.mean())
    return {
        "n_seeds": n,
        "eval_mean": mean,
        "eval_std": std,
        "ci95_lo": mean - 1.96 * sem,
        "ci95_hi": mean + 1.96 * sem,
        "wall": float(walls.mean()),
    }


def _print_results_summary(names: list[str]) -> None:
    """Per-experiment eval stats across seeds. Skips experiments with no results."""
    rows: list[tuple[str, dict]] = []
    for name in names:
        s = _collect_stats(name)
        if s is None:
            print(f"  (no results on disk for {name})")
        else:
            rows.append((name, s))
    if not rows:
        return

    name_w = max(len(name) for name, _ in rows)
    print(f"\n=== Results ({len(rows)} experiments) ===")
    header = (f"{'name':<{name_w}s}  {'seeds':>5s}  {'eval_mean':>10s}  "
              f"{'eval_std':>10s}  {'ci95_lo':>10s}  {'ci95_hi':>10s}  "
              f"{'wall':>8s}")
    print(header)
    print("-" * len(header))
    for name, s in rows:
        print(f"{name:<{name_w}s}  {s['n_seeds']:>5d}  "
              f"{s['eval_mean']:>10.4f}  {s['eval_std']:>10.4f}  "
              f"{s['ci95_lo']:>10.4f}  {s['ci95_hi']:>10.4f}  "
              f"{s['wall']:>8.3f}")


if __name__ == "__main__":
    sys.exit(main())
