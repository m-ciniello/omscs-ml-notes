"""Agent factory: `build_agent(spec)` -> ready agent.

Informal agent contract: every agent implements
    run(env, *, n_episodes, eval_episodes, gamma, seed) -> RunResult dict
with keys `eval_returns`, `eval_steps` (required), plus optional
`train_returns`, `train_steps`, `history`, `policy`, `Q`, `wall_clock_seconds`.
The runner tolerates missing optional fields.
"""

from __future__ import annotations

from typing import Any

from src.agents.dqn import DQNAgent
from src.agents.pi import PolicyIteration
from src.agents.q_learning import QLearning
from src.agents.random_agent import RandomAgent
from src.agents.sarsa import SARSA
from src.agents.vi import ValueIteration
from src.configs import AgentSpec


_REGISTRY: dict[str, type] = {
    "random": RandomAgent,
    "vi": ValueIteration,
    "pi": PolicyIteration,
    "sarsa": SARSA,
    "qlearning": QLearning,
    "dqn": DQNAgent,
}


def build_agent(spec: AgentSpec) -> Any:
    """Build an agent from a spec."""
    name = spec.name.lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown agent {spec.name!r}. "
            f"Registered: {sorted(_REGISTRY.keys())}"
        )
    return cls(**spec.hyperparams)


__all__ = ["build_agent"]
