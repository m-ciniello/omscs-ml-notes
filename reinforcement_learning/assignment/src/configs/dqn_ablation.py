"""Phase 4 — Rainbow-medium ablation on CartPole-v1 (extra credit).

Six variants sharing an identical base set of hyperparameters, differing
only in which Rainbow components are toggled on:
    baseline | +Double | +Dueling | +PER | +N-step | full Rainbow

C51 and NoisyNets are intentionally out of scope (time-boxed EC).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from src.configs._registry import (
    AgentSpec,
    EnvSpec,
    ExperimentSpec,
    register,
)


_SHARED_HP: dict[str, Any] = {
    "hidden": 128,
    "lr": 1e-3,
    "buffer_capacity": 10_000,
    "batch_size": 64,
    "warmup_steps": 500,
    "train_freq": 1,
    "target_update_freq": 500,
    "grad_clip": 10.0,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 10_000,
    # PER defaults (only read when per=True)
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
    "per_beta_steps": 20_000,
}

_BASE = ExperimentSpec(
    name="dqn_ablation_baseline",
    env=EnvSpec(name="cartpole_continuous"),
    agent=AgentSpec(name="dqn", hyperparams=dict(_SHARED_HP)),
    n_episodes=300,
    eval_episodes=20,
    seeds=tuple(range(10)),
    gamma=0.99,
    description="Vanilla DQN baseline on CartPole-v1 (continuous state, "
                "MLP Q-net). No Double / Dueling / PER / N-step. "
                "Reference bar for the ablation.",
    results_path_parts=("dqn_ablation", "baseline"),
)


def _register_variant(
    *,
    suffix: str,
    extra_hp: dict[str, Any],
    description: str,
) -> None:
    hp = dict(_SHARED_HP)
    hp.update(extra_hp)
    register(dataclasses.replace(
        _BASE,
        name=f"dqn_ablation_{suffix}",
        agent=AgentSpec(name="dqn", hyperparams=hp),
        description=description,
        results_path_parts=("dqn_ablation", suffix),
    ))


register(_BASE)

_register_variant(
    suffix="double",
    extra_hp={"double": True},
    description="DQN + Double-DQN: decouples action-selection (online net) "
                "from value-estimation (target net). Expected to reduce the "
                "positive bias of max-over-target that vanilla DQN suffers from.",
)

_register_variant(
    suffix="dueling",
    extra_hp={"dueling": True},
    description="DQN + Dueling network: V(s) + A(s,·) with mean-centered "
                "advantages. Mostly architectural — pays off more in "
                "environments where many actions have similar values.",
)

_register_variant(
    suffix="per",
    extra_hp={"per": True},
    description="DQN + Prioritized Experience Replay (proportional, sum-tree). "
                "Samples transitions with high TD-error more often; β anneals "
                "from 0.4 to 1.0 over 20k gradient steps to correct IS bias.",
)

_register_variant(
    suffix="nstep",
    extra_hp={"nstep": 3},
    description="DQN + 3-step TD targets. Trades a bit of bias (off-policy "
                "error from using the behaviour policy's multi-step returns) "
                "for lower variance and faster credit assignment.",
)

_register_variant(
    suffix="rainbow",
    extra_hp={"double": True, "dueling": True, "per": True, "nstep": 3},
    description="Rainbow-medium: Double + Dueling + PER + 3-step. The all-in "
                "variant. Comparing this against the four single-component "
                "variants isolates each component's marginal contribution.",
)
