"""Run all experiments matching a name prefix or tag, then summarise.

Typical usage:

    # Run every variant of a gamma sweep, then print the summary table
    python scripts/run_sweep.py --prefix blackjack_vi_gamma_sweep --sweep-path gamma

    # Run every DP-smoke experiment (tag filter)
    python scripts/run_sweep.py --tags smoke,dp

    # Skip running, just summarise existing results
    python scripts/run_sweep.py --prefix blackjack_vi_gamma_sweep --sweep-path gamma --no-run

This is the intended entry point for "reproduce an HP sweep" — one command,
deterministic outputs, summary printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.configs import list_experiments  # noqa: E402
from src.experiments.compare import load_sweep, print_sweep_summary  # noqa: E402
from src.experiments.runner import run_experiments  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Name prefix to match (e.g. blackjack_vi_gamma_sweep).",
    )
    p.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tags to match (experiments must carry ALL).",
    )
    p.add_argument(
        "--exclude-tags",
        type=str,
        default=None,
        help="Comma-separated tags to exclude (drop experiments with ANY).",
    )
    p.add_argument(
        "--sweep-path",
        type=str,
        default=None,
        help="Dotted path used at registration time. Required for summary "
             "output so per-variant HP values can be extracted from configs.",
    )
    p.add_argument(
        "--no-run",
        action="store_true",
        help="Skip running; only summarise existing results (must already exist).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if result.pkl already exists.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    tags = set(args.tags.split(",")) if args.tags else None
    exclude_tags = set(args.exclude_tags.split(",")) if args.exclude_tags else None

    names = list_experiments(
        name_prefix=args.prefix,
        tags=tags,
        exclude_tags=exclude_tags,
    )
    if not names:
        print("No experiments match the given filters. Registered matches:")
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
        sweep = load_sweep(names, sweep_path=args.sweep_path)
        print_sweep_summary(sweep)
    else:
        print("\n(no --sweep-path given; skipping sweep summary)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
