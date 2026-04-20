"""Phase 3 — Tabular SARSA / Q-Learning on discretized CartPole-v1.

CartPole's 4-D continuous obs are binned into a discrete state space.
Default grid (3,3,8,12) = 864 bins follows the FAQ starter; the n_bins
sweep extends that from very coarse (36 bins) to fine (4800 bins). Also
includes α and γ sweeps so each agent has ≥ 2 validated HPs on CartPole
(the FAQ minimum), with n_bins as a bonus third dimension for the
"discretization effect" rubric question.
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


# FAQ starter grid: angle / angular-velocity get finer resolution because
# they dominate the control problem; cart position / velocity are coarser.
CARTPOLE_DEFAULT_N_BINS = (3, 3, 8, 12)

_ENV = EnvSpec(
    name="cartpole",
    kwargs={"n_bins": CARTPOLE_DEFAULT_N_BINS},
)

_TABULAR_HP_BASE: dict[str, float | int] = {
    "alpha": 0.1,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 5_000,
    "max_steps_per_episode": 500,
}

_SARSA_BASE = ExperimentSpec(
    name="cartpole_sarsa_base",  # template; NOT registered directly
    env=_ENV,
    agent=AgentSpec(name="sarsa", hyperparams=dict(_TABULAR_HP_BASE)),
    n_episodes=10_000,
    eval_episodes=100,
    seeds=tuple(range(10)),
    gamma=0.99,
)

_QLEARNING_BASE = ExperimentSpec(
    name="cartpole_qlearning_base",  # template; NOT registered directly
    env=_ENV,
    agent=AgentSpec(name="qlearning", hyperparams=dict(_TABULAR_HP_BASE)),
    n_episodes=10_000,
    eval_episodes=100,
    seeds=tuple(range(10)),
    gamma=0.99,
)

# --- Reference single-point runs ------------------------------------------
register(dataclasses.replace(
    _SARSA_BASE, name="cartpole_sarsa_default",
    description="SARSA on discretized CartPole-v1 with (3,3,8,12) binning "
                "(FAQ starter grid; 864 bins). CartPole-v1 'solved' = 475; "
                "tabular typically reaches 150-350.",
))
register(dataclasses.replace(
    _QLEARNING_BASE, name="cartpole_qlearning_default",
    description="Q-learning on discretized CartPole-v1 with (3,3,8,12) "
                "binning; off-policy analogue of cartpole_sarsa_default.",
))

# --- HP sweeps ------------------------------------------------------------

# α sweep --------------------------------------------------------------
_ALPHA_VALUES = [0.05, 0.1, 0.2, 0.5]
register_sweep(
    name_prefix="cartpole_sarsa_alpha_sweep",
    base=_SARSA_BASE,
    sweep_path="agent.hyperparams.alpha",
    values=_ALPHA_VALUES,
    description="SARSA on CartPole: step-size sweep. Smaller α => more "
                "stable but slower; larger α => faster but noisier.",
)
register_sweep(
    name_prefix="cartpole_qlearning_alpha_sweep",
    base=_QLEARNING_BASE,
    sweep_path="agent.hyperparams.alpha",
    values=_ALPHA_VALUES,
    description="Q-Learning on CartPole: step-size sweep (off-policy twin).",
)

# γ sweep --------------------------------------------------------------
_GAMMA_VALUES = [0.9, 0.95, 0.99, 1.0]
register_sweep(
    name_prefix="cartpole_sarsa_gamma_sweep",
    base=_SARSA_BASE,
    sweep_path="gamma",
    values=_GAMMA_VALUES,
    description="SARSA on CartPole: discount sweep. CartPole's long "
                "horizon (up to 500 steps) makes γ first-class here, "
                "unlike Blackjack where γ is near-irrelevant.",
)
register_sweep(
    name_prefix="cartpole_qlearning_gamma_sweep",
    base=_QLEARNING_BASE,
    sweep_path="gamma",
    values=_GAMMA_VALUES,
    description="Q-Learning on CartPole: discount sweep (off-policy twin).",
)

# n_bins sweep ---------------------------------------------------------
# Grids span ~2 orders of magnitude in state count:
#   (1, 1, 6, 6)    =   36 states (cart state ignored entirely)
#   (3, 3, 6, 6)    =  324 states
#   (3, 3, 8, 12)   =  864 states  <- FAQ-recommended default
#   (5, 5, 12, 16)  = 4800 states  <- fine-grained
CARTPOLE_NBIN_GRIDS = [
    (1, 1, 6, 6),
    (3, 3, 6, 6),
    (3, 3, 8, 12),
    (5, 5, 12, 16),
]
register_sweep(
    name_prefix="cartpole_sarsa_nbins_sweep",
    base=_SARSA_BASE,
    sweep_path="env.kwargs.n_bins",
    values=CARTPOLE_NBIN_GRIDS,
    description="SARSA on CartPole: discretization sweep from 36 to 4800 "
                "bins. Expected bias-variance trade-off: coarse grids "
                "train fast but cap policy quality; fine grids need more "
                "samples to revisit each state.",
)
register_sweep(
    name_prefix="cartpole_qlearning_nbins_sweep",
    base=_QLEARNING_BASE,
    sweep_path="env.kwargs.n_bins",
    values=CARTPOLE_NBIN_GRIDS,
    description="Q-Learning on CartPole: discretization sweep "
                "(off-policy twin).",
)
