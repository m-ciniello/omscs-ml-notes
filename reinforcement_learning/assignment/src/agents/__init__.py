"""Agent factory.

Each registered agent name maps to a constructor that takes `hyperparams`.
The runner calls `build_agent(spec)` and trusts the factory to return a
ready agent.
"""

from __future__ import annotations

from src.configs import AgentSpec

from src.agents.base import BaseAgent
from src.agents.pi import PolicyIteration
from src.agents.q_learning import QLearning
from src.agents.random_agent import RandomAgent
from src.agents.sarsa import SARSA
from src.agents.vi import ValueIteration


_REGISTRY: dict[str, type[BaseAgent]] = {
    "random": RandomAgent,
    "vi": ValueIteration,
    "pi": PolicyIteration,
    "sarsa": SARSA,
    "qlearning": QLearning,
}


def build_agent(spec: AgentSpec) -> BaseAgent:
    """Build an agent from a spec.

    Args:
        spec: AgentSpec with `name` (registry key) and `hyperparams` (kwargs).

    Returns:
        A ready agent implementing `BaseAgent.run(env, ...)`.
    """
    name = spec.name.lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown agent {spec.name!r}. "
            f"Registered: {sorted(_REGISTRY.keys())}"
        )
    return cls(**spec.hyperparams)


__all__ = ["BaseAgent", "build_agent"]
