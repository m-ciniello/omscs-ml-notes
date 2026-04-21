"""Single-seed smoke test across all phases.

Exercises every code path (analytical MDP, estimated MDP, tabular, DQN)
before the 2-hour full campaign. Any failure here means a config or
registry bug; fix before launching the real run.

Writes results to `results/` — call with `rm -rf results/` afterwards
before the full campaign.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.configs import get
from src.experiments.runner import run_spec


SMOKES = [
    "blackjack_vi_default",
    "blackjack_pi_default",
    "blackjack_sarsa_default",
    "cartpole_sarsa_default",
    "cartpole_vi_nbins_sweep_1x1x6x6",
    "cartpole_pi_nbins_sweep_1x1x6x6",
    "dqn_ablation_baseline",
]


def main() -> int:
    failures = []
    for name in SMOKES:
        print(f"\n=== SMOKE: {name} ===")
        spec = get(name)
        one_seed = dataclasses.replace(spec, seeds=(0,))
        # DQN defaults to many episodes; cut for smoke
        if "dqn" in name and one_seed.n_episodes > 50:
            one_seed = dataclasses.replace(one_seed, n_episodes=50)
        t0 = time.perf_counter()
        try:
            run_spec(one_seed, overwrite=True, verbose=True)
            print(f"  -> OK in {time.perf_counter() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"  -> FAIL: {type(e).__name__}: {e}")
            failures.append((name, e))

    print()
    print("=" * 60)
    if failures:
        print(f"FAILED: {len(failures)}/{len(SMOKES)}")
        for name, err in failures:
            print(f"  {name}: {type(err).__name__}: {err}")
        return 1
    print(f"ALL {len(SMOKES)} SMOKES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
