"""Environment factory.

Each registered env name maps to a constructor. The runner calls
`build_env(spec, seed)` and trusts the factory to return a ready env.
"""

from __future__ import annotations

from typing import Any

from src.configs import EnvSpec

from src.envs.blackjack import Blackjack
from src.envs.cartpole import DiscretizedCartPole
from src.envs.gridworld import GridWorld


_REGISTRY: dict[str, Any] = {
    "gridworld": GridWorld,
    "blackjack": Blackjack,
    "cartpole": DiscretizedCartPole,
}


def build_env(spec: EnvSpec, seed: int | None = None) -> Any:
    """Build an environment from a spec.

    Args:
        spec: EnvSpec with `name` (registry key) and `kwargs` (constructor args).
        seed: seed forwarded to the env for reproducibility.

    Returns:
        A ready-to-use env exposing at minimum `reset()` and `step(action)`.
    """
    name = spec.name.lower()
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown env {spec.name!r}. "
            f"Registered: {sorted(_REGISTRY.keys())}"
        )
    return cls(seed=seed, **spec.kwargs)
