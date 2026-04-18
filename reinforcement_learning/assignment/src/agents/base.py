"""Common agent interface.

Every agent exposes a single `run(env, ...)` method that returns a
`RunResult` dict. This keeps the runner totally uniform across agent
types (DP, tabular, deep) — the runner doesn't care what "training" means
for a specific agent, it just asks the agent to run.

Agents should:
1. Train themselves inside `run` (for DP this is a solve; for model-free
   this is rollouts + updates; for DQN this is rollouts + gradient steps).
2. Evaluate their learned policy for `eval_episodes` rollouts.
3. Return a RunResult dict with the fields listed below.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np


class RunResult(TypedDict, total=False):
    """Canonical shape for agent outputs.

    Fields marked `total=False` aren't all required for every agent type —
    e.g. DP agents won't have `train_returns` since they don't train via
    rollouts. The aggregator tolerates missing fields.
    """

    # Training-time metrics (one entry per training episode, empty for DP).
    train_returns: list[float]
    train_steps: list[int]

    # Evaluation rollouts of the final learned policy.
    eval_returns: list[float]
    eval_steps: list[int]

    # Agent-specific diagnostics: convergence history, losses, etc.
    # Anything in here gets pickled alongside the result. Keep it JSON-able
    # where possible so the sidecar summary can include it.
    history: dict[str, Any]

    # Learned artefacts. `policy` is state -> action; `Q` is state -> list of
    # action-values (optional — DP fills both, Q-learning fills both, random
    # agent fills neither).
    policy: dict | np.ndarray | None
    Q: dict | np.ndarray | None

    # Resource usage.
    wall_clock_seconds: float


class BaseAgent:
    """Minimal contract every agent must satisfy.

    Agents should not access randomness through the global `random` module;
    instead, derive per-call RNGs from the seed passed to `run`. This keeps
    runs bit-exact reproducible across machines.
    """

    name: str = "base"

    def run(
        self,
        env,
        *,
        n_episodes: int,
        eval_episodes: int,
        gamma: float,
        seed: int,
    ) -> RunResult:
        """Train the agent on `env`, evaluate it, and return a RunResult."""
        raise NotImplementedError
