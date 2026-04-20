"""Phase 2 — Tabular model-free (SARSA, Q-Learning) on Blackjack.

Two reference runs (SARSA / Q-Learning at shared defaults) plus two HP
sweeps per algorithm (α step-size and ε-decay horizon). The DP optimum
(mean eval return ≈ -0.046) is the asymptotic target.
"""

from __future__ import annotations

import dataclasses

from src.configs._registry import (
    AgentSpec,
    EnvSpec,
    ExperimentSpec,
    register,
    register_sweep,
)


_TABULAR_HP_BASE: dict[str, float | int] = {
    "alpha": 0.05,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 100_000,
    "max_steps_per_episode": 50,  # Blackjack episodes almost never exceed ~10
}

_SARSA_BASE = ExperimentSpec(
    name="blackjack_sarsa_base",  # template; NOT registered directly
    env=EnvSpec(name="blackjack"),
    agent=AgentSpec(name="sarsa", hyperparams=dict(_TABULAR_HP_BASE)),
    n_episodes=200_000,
    eval_episodes=20_000,
    seeds=tuple(range(10)),
    gamma=1.0,
)

_QLEARNING_BASE = ExperimentSpec(
    name="blackjack_qlearning_base",  # template; NOT registered directly
    env=EnvSpec(name="blackjack"),
    agent=AgentSpec(name="qlearning", hyperparams=dict(_TABULAR_HP_BASE)),
    n_episodes=200_000,
    eval_episodes=20_000,
    seeds=tuple(range(10)),
    gamma=1.0,
)

# --- Reference single-point runs ------------------------------------------
register(dataclasses.replace(
    _SARSA_BASE, name="blackjack_sarsa_default",
    description="SARSA on Blackjack; target is to approach the DP optimum "
                "(eval_return ≈ -0.046) via on-policy TD(0).",
))
register(dataclasses.replace(
    _QLEARNING_BASE, name="blackjack_qlearning_default",
    description="Q-learning on Blackjack; off-policy analogue of "
                "blackjack_sarsa_default with an identical HP configuration.",
))

# --- HP sweeps (α and ε-decay horizon, shared across both agents) --------
_ALPHA_VALUES = [0.01, 0.05, 0.1, 0.2]
_EPS_DECAY_VALUES = [10_000, 50_000, 100_000, 200_000]

register_sweep(
    name_prefix="blackjack_sarsa_alpha_sweep",
    base=_SARSA_BASE,
    sweep_path="agent.hyperparams.alpha",
    values=_ALPHA_VALUES,
    description="SARSA on Blackjack: step-size / stability tradeoff.",
)
register_sweep(
    name_prefix="blackjack_qlearning_alpha_sweep",
    base=_QLEARNING_BASE,
    sweep_path="agent.hyperparams.alpha",
    values=_ALPHA_VALUES,
    description="Q-learning on Blackjack: step-size / stability tradeoff.",
)

register_sweep(
    name_prefix="blackjack_sarsa_eps_decay_sweep",
    base=_SARSA_BASE,
    sweep_path="agent.hyperparams.epsilon_decay_episodes",
    values=_EPS_DECAY_VALUES,
    description="SARSA on Blackjack: how long should ε decay run? The FAQ "
                "warns 'ε decays to near-zero too early ⇒ premature "
                "exploitation'; this sweep directly tests that failure mode.",
)
register_sweep(
    name_prefix="blackjack_qlearning_eps_decay_sweep",
    base=_QLEARNING_BASE,
    sweep_path="agent.hyperparams.epsilon_decay_episodes",
    values=_EPS_DECAY_VALUES,
    description="Q-learning on Blackjack: how long should ε decay run?",
)
