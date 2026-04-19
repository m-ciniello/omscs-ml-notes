"""Run every experiment matching a name prefix or tag, then summarise.

    python scripts/run_sweep.py --prefix blackjack_vi_gamma_sweep --sweep-path gamma
    python scripts/run_sweep.py --tags smoke,dp
    python scripts/run_sweep.py --prefix blackjack_vi_gamma_sweep --sweep-path gamma --no-run

Deterministic outputs, summary printed. This is the intended reproduce-an-HP-sweep
entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.configs import list_experiments  # noqa: E402
from src.experiments.aggregate import load_config, load_runs  # noqa: E402
from src.experiments.runner import run_experiments  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefix", type=str, default=None,
                   help="Name prefix (e.g. blackjack_vi_gamma_sweep).")
    p.add_argument("--tags", type=str, default=None,
                   help="Comma-separated tags (experiments must carry ALL).")
    p.add_argument("--exclude-tags", type=str, default=None,
                   help="Comma-separated tags to exclude.")
    p.add_argument("--sweep-path", type=str, default=None,
                   help="Dotted path into the spec to extract each variant's "
                        "HP value. Required for the summary table.")
    p.add_argument("--no-run", action="store_true",
                   help="Skip running; only summarise existing results.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if result.pkl already exists.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    tags = set(args.tags.split(",")) if args.tags else None
    exclude_tags = set(args.exclude_tags.split(",")) if args.exclude_tags else None

    names = list_experiments(name_prefix=args.prefix, tags=tags,
                             exclude_tags=exclude_tags)
    if not names:
        print("No experiments match the given filters. Registered:")
        for n in list_experiments():
            print(f"  {n}")
        return 1

    print(f"Matched {len(names)} experiments:")
    for n in names:
        print(f"  {n}")

    if not args.no_run:
        print("\n--- Running ---")
        run_experiments(names, overwrite=args.overwrite)

    if args.sweep_path:
        print("\n--- Summary ---")
        _print_sweep_summary(names, args.sweep_path)
    else:
        print("\n(no --sweep-path given; skipping sweep summary)")

    return 0


def _print_sweep_summary(names: list[str], sweep_path: str) -> None:
    """For each variant, mean +/- 1.96 * SE of eval_return_mean across seeds."""
    rows: list[tuple] = []
    for name in names:
        cfg = load_config(name)
        value = _read_path(cfg, sweep_path)
        if isinstance(value, list):
            value = tuple(value)
        runs = load_runs(name)
        vals = np.array([r.summary.get("eval_return_mean", np.nan)
                         for r in runs], dtype=float)
        walls = np.array([r.summary.get("wall_clock_seconds", 0.0)
                          for r in runs], dtype=float)
        n = len(vals)
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        mean = float(vals.mean())
        rows.append((value, mean, std,
                     mean - 1.96 * sem, mean + 1.96 * sem,
                     float(walls.mean())))

    # Sort numerically when possible, else lexicographically.
    def _sort_key(row):
        try:
            return (0, float(row[0]))
        except (TypeError, ValueError):
            return (1, str(row[0]))
    rows.sort(key=_sort_key)

    print(f"\n=== Sweep over {sweep_path} ({len(rows)} values) ===")
    header = (f"{'value':>10s}  {'eval_mean':>10s}  {'eval_std':>10s}  "
              f"{'ci95_lo':>10s}  {'ci95_hi':>10s}  {'wall':>8s}")
    print(header)
    print("-" * len(header))
    for value, mean, std, lo, hi, wall in rows:
        v_str = f"{value:g}" if isinstance(value, float) else str(value)
        print(f"{v_str:>10s}  {mean:>10.4f}  {std:>10.4f}  "
              f"{lo:>10.4f}  {hi:>10.4f}  {wall:>8.3f}")


def _read_path(obj: dict, path: str):
    """Walk a dotted path into a nested dict (from asdict(ExperimentSpec))."""
    cur = obj
    for part in path.split("."):
        cur = cur[part]
    return cur


if __name__ == "__main__":
    sys.exit(main())
