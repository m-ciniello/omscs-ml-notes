"""Single source of truth for experiments in this assignment.

Every experiment is a named entry in the `EXPERIMENTS` registry below.
Reproducing a result means running the registered experiment by name;
the runner reads the spec here, builds the env + agent, runs across seeds,
and writes a config snapshot alongside every result dump.

This module is intentionally pure data. Factory logic (how to build an
env or agent from a spec) lives in `src/envs/__init__.py` and
`src/agents/__init__.py`.

Organisation:
    - Spec dataclasses (EnvSpec, AgentSpec, ExperimentSpec)
    - Registry + lookup helpers (register, get, list_experiments)
    - Sweep generator (register_sweep) — the main ergonomic affordance
      for hyperparameter studies
    - Registered experiments, grouped by phase

Naming convention for experiments:
    <env>_<agent>_<variant>              # single-point configs
    <env>_<agent>_<hp>_sweep_<value>     # sweep variants, same prefix
                                         # identifies a sweep for grouping

Tags supplement names for cross-cutting filtering. Every sweep variant
carries the tags "sweep" and "sweep:<path>" so they can be identified
regardless of name.

Results directory layout:
    Each spec carries `results_path_parts` (a path relative to results/).
    - Standalone runs: ("<name>",)              -> results/<name>/
    - Sweep variants:  ("<prefix>", "<frag>")   -> results/<prefix>/<frag>/
    If empty (the default for manually-registered specs), the runner falls
    back to ("<name>",) so existing single-point specs don't need edits.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnvSpec:
    """How to build an environment."""
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSpec:
    """How to build an agent."""
    name: str
    hyperparams: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    """A full experiment: env + agent + training budget + seeds.

    Attributes:
        name: unique identifier, used as the results subdirectory name.
        env: environment spec.
        agent: agent spec.
        n_episodes: training budget for model-free agents. Ignored by DP.
        eval_episodes: how many rollouts to average for final-performance stats.
        seeds: one run per seed; all reported metrics are aggregated across seeds.
        gamma: discount factor (shared across agent and env where relevant).
        tags: free-form labels for filtering / grouping in analysis.
        description: human-readable one-liner summarising the experiment.
        results_path_parts: path components (relative to results/) where this
            experiment's outputs live. If empty, the runner falls back to
            ("<name>",). Sweep variants set this to ("<prefix>", "<frag>")
            so all variants of a sweep nest under a shared parent directory.
    """
    name: str
    env: EnvSpec
    agent: AgentSpec
    n_episodes: int = 5000
    eval_episodes: int = 100
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    gamma: float = 0.99
    tags: tuple[str, ...] = ()
    description: str = ""
    results_path_parts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Registry + lookup
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[str, ExperimentSpec] = {}


def register(exp: ExperimentSpec) -> ExperimentSpec:
    """Add an experiment to the global registry. Duplicate names are errors."""
    if exp.name in EXPERIMENTS:
        raise ValueError(f"Experiment {exp.name!r} already registered")
    EXPERIMENTS[exp.name] = exp
    return exp


def get(name: str) -> ExperimentSpec:
    """Look up an experiment spec by name with a helpful error on misses."""
    if name not in EXPERIMENTS:
        raise KeyError(
            f"No experiment named {name!r}. "
            f"Registered: {sorted(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[name]


def list_experiments(
    name_prefix: str | None = None,
    tags: set[str] | None = None,
    exclude_tags: set[str] | None = None,
) -> list[str]:
    """List registered experiment names matching given criteria.

    Args:
        name_prefix: keep only names starting with this prefix.
        tags: keep only experiments whose tags are a superset of this set.
        exclude_tags: drop experiments carrying any of these tags.

    Returns:
        Sorted list of matching experiment names.
    """
    out: list[str] = []
    for name, spec in EXPERIMENTS.items():
        if name_prefix is not None and not name.startswith(name_prefix):
            continue
        spec_tags = set(spec.tags)
        if tags is not None and not tags.issubset(spec_tags):
            continue
        if exclude_tags is not None and spec_tags & exclude_tags:
            continue
        out.append(name)
    return sorted(out)


# ---------------------------------------------------------------------------
# Sweep generator
# ---------------------------------------------------------------------------

def register_sweep(
    *,
    name_prefix: str,
    base: ExperimentSpec,
    sweep_path: str,
    values: list,
    extra_tags: tuple[str, ...] = (),
    description: str | None = None,
) -> list[ExperimentSpec]:
    """Register one variant per sweep value.

    Each variant is a copy of `base` with the given sweep_path overridden
    and a unique name `{name_prefix}_{value_str}`. All variants share the
    tags `("sweep", f"sweep:{sweep_path}", name_prefix, ...base.tags)`,
    which lets the compare tooling collect a full sweep by tag.

    Args:
        name_prefix: common prefix for every variant's name, also added as a
            tag so the full sweep can be looked up by a single identifier.
        base: template ExperimentSpec. Its `.name` is ignored (each variant
            gets its own generated name) but everything else is copied.
        sweep_path: dotted path into the spec identifying which field to
            override. Supported paths:
                - "gamma" / "n_episodes" / "eval_episodes"  (ExperimentSpec field)
                - "agent.hyperparams.<key>"                  (nested hyperparam)
        values: list of values to try. Each must be JSON-serialisable.
        extra_tags: optional additional tags to apply to every variant.
        description: optional override; defaults to describing the sweep.

    Returns:
        List of registered ExperimentSpec objects (one per value).
    """
    sweep_tags = tuple(dict.fromkeys(  # dedup while preserving order
        base.tags + extra_tags + ("sweep", f"sweep:{sweep_path}", name_prefix)
    ))
    if description is None:
        description = (
            f"Sweep over {sweep_path} in {list(values)!r}. "
            f"Base: env={base.env.name}, agent={base.agent.name}."
        )

    variants: list[ExperimentSpec] = []
    for value in values:
        value_frag = _fmt_value(value)
        variant_name = f"{name_prefix}_{value_frag}"
        variant = _override_at_path(
            base,
            sweep_path=sweep_path,
            value=value,
            new_name=variant_name,
            new_tags=sweep_tags,
            new_description=description,
            new_results_path_parts=(name_prefix, value_frag),
        )
        register(variant)
        variants.append(variant)
    return variants


# ---------------------------------------------------------------------------
# Internals for register_sweep
# ---------------------------------------------------------------------------

def _fmt_value(v: Any) -> str:
    """Format a sweep value as a safe identifier fragment.

    Examples:
        0.95       -> "0p95"
        1.0        -> "1p0"
        1e-09      -> "1e-09"        (scientific notation has no dots)
        200        -> "200"
        "relu"     -> "relu"
    """
    if isinstance(v, float):
        s = repr(v)  # preserves precision; "0.95" not "0.949999..."
        if "e" in s or "E" in s:
            return s
        return s.replace(".", "p").replace("-", "m")
    return str(v)


def _override_at_path(
    base: ExperimentSpec,
    *,
    sweep_path: str,
    value: Any,
    new_name: str,
    new_tags: tuple[str, ...],
    new_description: str,
    new_results_path_parts: tuple[str, ...],
) -> ExperimentSpec:
    """Return a copy of base with value applied at sweep_path."""
    parts = sweep_path.split(".")

    # Case 1: "agent.hyperparams.<key>"
    if len(parts) == 3 and parts[0] == "agent" and parts[1] == "hyperparams":
        key = parts[2]
        new_hp = dict(base.agent.hyperparams)
        new_hp[key] = value
        new_agent = dataclasses.replace(base.agent, hyperparams=new_hp)
        return dataclasses.replace(
            base,
            name=new_name,
            agent=new_agent,
            tags=new_tags,
            description=new_description,
            results_path_parts=new_results_path_parts,
        )

    # Case 2: top-level ExperimentSpec field ("gamma", "n_episodes", ...)
    if len(parts) == 1:
        field_name = parts[0]
        if field_name not in {"gamma", "n_episodes", "eval_episodes", "seeds"}:
            raise ValueError(
                f"sweep_path={sweep_path!r} refers to an ExperimentSpec field "
                f"that isn't safe to sweep over. Allowed: gamma, n_episodes, "
                f"eval_episodes, seeds."
            )
        return dataclasses.replace(
            base,
            name=new_name,
            tags=new_tags,
            description=new_description,
            results_path_parts=new_results_path_parts,
            **{field_name: value},
        )

    raise ValueError(
        f"Unsupported sweep_path {sweep_path!r}. Use 'gamma' or "
        f"'agent.hyperparams.<key>'."
    )


# ===========================================================================
# Registered experiments
# ===========================================================================

# --- Phase 0: smoke tests ---------------------------------------------------

register(ExperimentSpec(
    name="smoke_gridworld_random",
    env=EnvSpec(name="gridworld", kwargs={"rows": 3, "cols": 3}),
    agent=AgentSpec(name="random", hyperparams={}),
    n_episodes=50,
    eval_episodes=20,
    seeds=(0, 1, 2),
    tags=("smoke", "gridworld", "random"),
    description="End-to-end harness smoke test on a 3x3 gridworld with a random agent.",
))

register(ExperimentSpec(
    name="smoke_gridworld_vi",
    env=EnvSpec(name="gridworld", kwargs={"rows": 5, "cols": 5}),
    agent=AgentSpec(name="vi", hyperparams={"theta": 1e-8, "max_sweeps": 200}),
    n_episodes=0,
    eval_episodes=50,
    seeds=(0, 1, 2),
    gamma=0.95,
    tags=("smoke", "gridworld", "vi", "dp"),
    description="Sanity check: VI should solve 5x5 gridworld to optimality.",
))

register(ExperimentSpec(
    name="smoke_gridworld_pi",
    env=EnvSpec(name="gridworld", kwargs={"rows": 5, "cols": 5}),
    agent=AgentSpec(name="pi", hyperparams={
        "theta": 1e-8,
        "eval_max_sweeps": 200,
        "max_outer_iters": 50,
    }),
    n_episodes=0,
    eval_episodes=50,
    seeds=(0, 1, 2),
    gamma=0.95,
    tags=("smoke", "gridworld", "pi", "dp"),
    description="Sanity check: PI should match VI on 5x5 gridworld.",
))

register(ExperimentSpec(
    name="smoke_gridworld_sarsa",
    env=EnvSpec(name="gridworld", kwargs={"rows": 5, "cols": 5}),
    agent=AgentSpec(name="sarsa", hyperparams={
        "alpha": 0.1,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay_episodes": 1_000,
        "max_steps_per_episode": 200,
    }),
    n_episodes=2_000,
    eval_episodes=50,
    seeds=(0, 1, 2),
    gamma=0.95,
    tags=("smoke", "gridworld", "sarsa", "tabular"),
    description="Sanity check: SARSA should approach VI's optimal (-7) on 5x5 gridworld.",
))

register(ExperimentSpec(
    name="smoke_gridworld_qlearning",
    env=EnvSpec(name="gridworld", kwargs={"rows": 5, "cols": 5}),
    agent=AgentSpec(name="qlearning", hyperparams={
        "alpha": 0.1,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay_episodes": 1_000,
        "max_steps_per_episode": 200,
    }),
    n_episodes=2_000,
    eval_episodes=50,
    seeds=(0, 1, 2),
    gamma=0.95,
    tags=("smoke", "gridworld", "qlearning", "tabular"),
    description="Sanity check: Q-learning should match VI's optimal (-7) on 5x5 gridworld.",
))

# --- Phase 1: Blackjack DP --------------------------------------------------
# Each algorithm has a "default" reference run (tight theta, gamma=1) plus
# two HP sweeps (gamma and theta) that supply the variability data required
# by the assignment FAQ.

_BLACKJACK_VI_BASE = ExperimentSpec(
    name="blackjack_vi_base",  # template name; this spec is NOT registered directly
    env=EnvSpec(name="blackjack"),
    agent=AgentSpec(name="vi", hyperparams={"theta": 1e-9, "max_sweeps": 500}),
    n_episodes=0,
    eval_episodes=20_000,
    seeds=(0, 1, 2, 3, 4),
    gamma=1.0,
    tags=("blackjack", "vi", "dp"),
)

_BLACKJACK_PI_BASE = ExperimentSpec(
    name="blackjack_pi_base",  # template name; NOT registered directly
    env=EnvSpec(name="blackjack"),
    agent=AgentSpec(name="pi", hyperparams={
        "theta": 1e-9,
        "eval_max_sweeps": 500,
        "max_outer_iters": 50,
    }),
    n_episodes=0,
    eval_episodes=20_000,
    seeds=(0, 1, 2, 3, 4),
    gamma=1.0,
    tags=("blackjack", "pi", "dp"),
)

# Default single-point runs (pinned at "best" settings — tight theta, gamma=1).
register(dataclasses.replace(_BLACKJACK_VI_BASE, name="blackjack_vi_default",
         description="VI on the analytical Blackjack MDP, evaluated via 20k Gym rollouts."))
register(dataclasses.replace(_BLACKJACK_PI_BASE, name="blackjack_pi_default",
         description="PI on the analytical Blackjack MDP. Pair with blackjack_vi_default "
                     "for a convergence-speed comparison."))

# --- Phase 1c: HP sweeps ----------------------------------------------------
# Gamma sweep (discount factor). Blackjack is episodic with bounded horizon,
# so gamma shouldn't change optimal policy for reachable states — but it
# affects convergence speed and numerical behaviour in interesting ways.
_GAMMA_VALUES = [0.8, 0.9, 0.95, 0.99, 1.0]

register_sweep(
    name_prefix="blackjack_vi_gamma_sweep",
    base=_BLACKJACK_VI_BASE,
    sweep_path="gamma",
    values=_GAMMA_VALUES,
    description="VI on Blackjack: effect of discount factor on convergence "
                "speed and final policy quality.",
)

register_sweep(
    name_prefix="blackjack_pi_gamma_sweep",
    base=_BLACKJACK_PI_BASE,
    sweep_path="gamma",
    values=_GAMMA_VALUES,
    description="PI on Blackjack: effect of discount factor.",
)

# Theta sweep (convergence tolerance). Shows the precision / wall-clock
# tradeoff: the looser theta, the fewer sweeps, but past a problem-dependent
# threshold the greedy policy stabilises regardless.
_THETA_VALUES = [1e-1, 1e-3, 1e-5, 1e-9]

register_sweep(
    name_prefix="blackjack_vi_theta_sweep",
    base=_BLACKJACK_VI_BASE,
    sweep_path="agent.hyperparams.theta",
    values=_THETA_VALUES,
    description="VI on Blackjack: how tight does the Bellman residual need "
                "to be before the greedy policy stops changing?",
)

register_sweep(
    name_prefix="blackjack_pi_theta_sweep",
    base=_BLACKJACK_PI_BASE,
    sweep_path="agent.hyperparams.theta",
    values=_THETA_VALUES,
    description="PI on Blackjack: policy-evaluation precision vs wall clock.",
)

# --- Phase 2: tabular model-free on Blackjack -------------------------------
# Reference runs at reasonable defaults. The DP optimum is eval_return ≈ -0.046
# (mean over 5 seeds × 20k Gym rollouts); SARSA / Q-learning after 200k
# training episodes should come within ~0.005 of that.

_BLACKJACK_TABULAR_HP_BASE: dict[str, float | int] = {
    "alpha": 0.05,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 100_000,
    "max_steps_per_episode": 50,  # Blackjack episodes almost never exceed ~10
}

_BLACKJACK_SARSA_BASE = ExperimentSpec(
    name="blackjack_sarsa_base",  # template; NOT registered directly
    env=EnvSpec(name="blackjack"),
    agent=AgentSpec(name="sarsa", hyperparams=dict(_BLACKJACK_TABULAR_HP_BASE)),
    n_episodes=200_000,
    eval_episodes=20_000,
    seeds=(0, 1, 2, 3, 4),
    gamma=1.0,
    tags=("blackjack", "sarsa", "tabular"),
)

_BLACKJACK_QLEARNING_BASE = ExperimentSpec(
    name="blackjack_qlearning_base",  # template; NOT registered directly
    env=EnvSpec(name="blackjack"),
    agent=AgentSpec(name="qlearning", hyperparams=dict(_BLACKJACK_TABULAR_HP_BASE)),
    n_episodes=200_000,
    eval_episodes=20_000,
    seeds=(0, 1, 2, 3, 4),
    gamma=1.0,
    tags=("blackjack", "qlearning", "tabular"),
)

register(dataclasses.replace(_BLACKJACK_SARSA_BASE, name="blackjack_sarsa_default",
         description="SARSA on Blackjack; target is to approach the DP optimum "
                     "(eval_return ≈ -0.046) via on-policy TD(0)."))
register(dataclasses.replace(_BLACKJACK_QLEARNING_BASE, name="blackjack_qlearning_default",
         description="Q-learning on Blackjack; off-policy analogue of "
                     "blackjack_sarsa_default with an identical HP configuration."))

# A shorter ε-decay base used only for the sample-complexity (n_episodes)
# sweep. Fixing decay at 10k means every variant — even the smallest —
# completes its exploration schedule well before training ends, so the
# resulting eval-return curve is a clean function of training budget
# rather than a confound of "ran out of time mid-decay."
_BLACKJACK_TABULAR_BUDGET_HP: dict[str, float | int] = {
    **_BLACKJACK_TABULAR_HP_BASE,
    "epsilon_decay_episodes": 10_000,
}

_BLACKJACK_SARSA_BUDGET_BASE = dataclasses.replace(
    _BLACKJACK_SARSA_BASE,
    agent=AgentSpec(name="sarsa", hyperparams=dict(_BLACKJACK_TABULAR_BUDGET_HP)),
)
_BLACKJACK_QLEARNING_BUDGET_BASE = dataclasses.replace(
    _BLACKJACK_QLEARNING_BASE,
    agent=AgentSpec(name="qlearning", hyperparams=dict(_BLACKJACK_TABULAR_BUDGET_HP)),
)

# --- Phase 2b: tabular HP sweeps -------------------------------------------

_ALPHA_VALUES = [0.01, 0.05, 0.1, 0.2]
_EPS_DECAY_VALUES = [10_000, 50_000, 100_000, 200_000]
_NEPISODES_VALUES = [20_000, 100_000, 500_000]

register_sweep(
    name_prefix="blackjack_sarsa_alpha_sweep",
    base=_BLACKJACK_SARSA_BASE,
    sweep_path="agent.hyperparams.alpha",
    values=_ALPHA_VALUES,
    description="SARSA on Blackjack: step-size / stability tradeoff.",
)

register_sweep(
    name_prefix="blackjack_qlearning_alpha_sweep",
    base=_BLACKJACK_QLEARNING_BASE,
    sweep_path="agent.hyperparams.alpha",
    values=_ALPHA_VALUES,
    description="Q-learning on Blackjack: step-size / stability tradeoff.",
)

register_sweep(
    name_prefix="blackjack_sarsa_eps_decay_sweep",
    base=_BLACKJACK_SARSA_BASE,
    sweep_path="agent.hyperparams.epsilon_decay_episodes",
    values=_EPS_DECAY_VALUES,
    description="SARSA on Blackjack: how long should ε decay run?",
)

register_sweep(
    name_prefix="blackjack_qlearning_eps_decay_sweep",
    base=_BLACKJACK_QLEARNING_BASE,
    sweep_path="agent.hyperparams.epsilon_decay_episodes",
    values=_EPS_DECAY_VALUES,
    description="Q-learning on Blackjack: how long should ε decay run?",
)

register_sweep(
    name_prefix="blackjack_sarsa_nepisodes_sweep",
    base=_BLACKJACK_SARSA_BUDGET_BASE,
    sweep_path="n_episodes",
    values=_NEPISODES_VALUES,
    description="SARSA on Blackjack: sample-complexity curve. Uses a short "
                "ε-decay (10k) so every variant finishes its exploration "
                "schedule before training ends.",
)

register_sweep(
    name_prefix="blackjack_qlearning_nepisodes_sweep",
    base=_BLACKJACK_QLEARNING_BUDGET_BASE,
    sweep_path="n_episodes",
    values=_NEPISODES_VALUES,
    description="Q-learning on Blackjack: sample-complexity curve.",
)

# --- Phase 3: CartPole (the rollout-only "second environment") --------------
# CartPole's 4-D continuous observations are binned into 3x3x6x6 = 324
# states. No analytical MDP is exposed — this env is deliberately the
# "rollout-only" foil to Blackjack's analytical-MDP side. Compare
# tabular results here with the DP optimum on Blackjack to see the
# difference model-free makes when no model is available.
#
# Reward is +1 per step, truncated at 500. Random policy returns ~20;
# a well-tuned tabular agent typically reaches 100-400 at this
# discretization. "Solved" (CartPole-v1 convention) is mean >= 475
# over 100 eval episodes, which tabular+binning usually can't hit —
# that ceiling motivates the DQN work in Phase 4.

_CARTPOLE_ENV = EnvSpec(
    name="cartpole",
    kwargs={
        # Default (3, 3, 6, 6) binning lives on the env side; we override
        # here only where an experiment needs different granularity.
    },
)

# Smoke tests: very short runs that should still clearly beat random.
register(ExperimentSpec(
    name="smoke_cartpole_random",
    env=_CARTPOLE_ENV,
    agent=AgentSpec(name="random", hyperparams={"max_steps_per_episode": 500}),
    n_episodes=100,
    eval_episodes=50,
    seeds=(0, 1, 2),
    tags=("smoke", "cartpole", "random"),
    description="Random-policy baseline on CartPole; expected eval return ~20-40.",
))

register(ExperimentSpec(
    name="smoke_cartpole_qlearning",
    env=_CARTPOLE_ENV,
    agent=AgentSpec(name="qlearning", hyperparams={
        "alpha": 0.1,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_episodes": 1_000,
        "max_steps_per_episode": 500,
    }),
    n_episodes=2_000,
    eval_episodes=50,
    seeds=(0, 1, 2),
    gamma=0.99,
    tags=("smoke", "cartpole", "qlearning", "tabular"),
    description="Smoke: Q-learning on discretized CartPole; should clearly beat random.",
))

# Defaults: longer training for real eval numbers.
_CARTPOLE_TABULAR_HP_BASE: dict[str, float | int] = {
    "alpha": 0.1,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_episodes": 5_000,
    "max_steps_per_episode": 500,
}

_CARTPOLE_SARSA_BASE = ExperimentSpec(
    name="cartpole_sarsa_base",  # template; NOT registered directly
    env=_CARTPOLE_ENV,
    agent=AgentSpec(name="sarsa", hyperparams=dict(_CARTPOLE_TABULAR_HP_BASE)),
    n_episodes=10_000,
    eval_episodes=100,
    seeds=(0, 1, 2, 3, 4),
    gamma=0.99,
    tags=("cartpole", "sarsa", "tabular"),
)

_CARTPOLE_QLEARNING_BASE = ExperimentSpec(
    name="cartpole_qlearning_base",
    env=_CARTPOLE_ENV,
    agent=AgentSpec(name="qlearning", hyperparams=dict(_CARTPOLE_TABULAR_HP_BASE)),
    n_episodes=10_000,
    eval_episodes=100,
    seeds=(0, 1, 2, 3, 4),
    gamma=0.99,
    tags=("cartpole", "qlearning", "tabular"),
)

register(dataclasses.replace(_CARTPOLE_SARSA_BASE, name="cartpole_sarsa_default",
         description="SARSA on discretized CartPole-v1 with 3x3x6x6 binning. "
                     "CartPole-v1 solved = 475; tabular aims for >= 150."))
register(dataclasses.replace(_CARTPOLE_QLEARNING_BASE, name="cartpole_qlearning_default",
         description="Q-learning on discretized CartPole-v1; off-policy "
                     "analogue of cartpole_sarsa_default."))
