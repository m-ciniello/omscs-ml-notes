"""Experiment registry.

Public surface (used by scripts, runner, figure code):
    ExperimentSpec, EnvSpec, AgentSpec
    EXPERIMENTS, register, register_sweep, get, list_experiments
    override_at_path, _fmt_value (used by make_figures.py)

To add a new experiment: pick the topic module that fits (or add a new
one), call `register()` / `register_sweep()` in it, and import it below
so its side effects fire at package import time.
"""

from src.configs._registry import (
    AgentSpec,
    EnvSpec,
    EXPERIMENTS,
    ExperimentSpec,
    _fmt_value,
    get,
    list_experiments,
    override_at_path,
    register,
    register_sweep,
)

from src.configs import (  # noqa: F401  (imported for registration side effects)
    blackjack_dp,
    blackjack_tabular,
    cartpole_tabular,
    cartpole_dp,
    dqn_ablation,
)


__all__ = [
    "AgentSpec",
    "EnvSpec",
    "ExperimentSpec",
    "EXPERIMENTS",
    "_fmt_value",
    "get",
    "list_experiments",
    "override_at_path",
    "register",
    "register_sweep",
]
