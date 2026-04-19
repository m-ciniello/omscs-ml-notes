"""Environment factory: `build_env(spec, seed)` -> ready env."""

from __future__ import annotations

from typing import Any

from src.configs import EnvSpec
from src.envs.blackjack import Blackjack
from src.envs.cartpole import ContinuousCartPole, DiscretizedCartPole
from src.envs.cartpole_mdp import CartPoleEstimatedMDP


_REGISTRY: dict[str, Any] = {
    "blackjack": Blackjack,
    "cartpole": DiscretizedCartPole,
    "cartpole_continuous": ContinuousCartPole,
    "cartpole_estimated": CartPoleEstimatedMDP,
}


def build_env(spec: EnvSpec, seed: int | None = None) -> Any:
    cls = _REGISTRY.get(spec.name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown env {spec.name!r}. Registered: {sorted(_REGISTRY.keys())}"
        )
    return cls(seed=seed, **spec.kwargs)
