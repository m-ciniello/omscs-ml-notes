"""Single source of truth for experiments.

Every experiment is a named entry in the EXPERIMENTS registry. Reproducing
a result means running the registered experiment by name; the runner
snapshots the spec verbatim next to every result dump.

Naming:
    <env>_<agent>_<variant>              # single-point
    <env>_<agent>_<hp>_sweep_<value>     # shared-prefix sweep variants

Sweep variants live under `results/<prefix>/<frag>/`; single-point specs
use `results/<name>/`. All driven by ExperimentSpec.results_path_parts.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any


# --- Spec dataclasses ---

@dataclass(frozen=True)
class EnvSpec:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    hyperparams: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSpec:
    """env + agent + training budget + seeds.

    `results_path_parts` drives the on-disk location: empty -> results/<name>/,
    else results/<parts...>/. Sweep variants use ("<prefix>", "<frag>") so
    every variant of a sweep nests under a shared parent.
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


# --- Registry + lookup ---

EXPERIMENTS: dict[str, ExperimentSpec] = {}


def register(exp: ExperimentSpec) -> ExperimentSpec:
    if exp.name in EXPERIMENTS:
        raise ValueError(f"Experiment {exp.name!r} already registered")
    EXPERIMENTS[exp.name] = exp
    return exp


def get(name: str) -> ExperimentSpec:
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
    """Filtered + sorted list of registered experiment names."""
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


# --- Sweep generator ---

def register_sweep(
    *,
    name_prefix: str,
    base: ExperimentSpec,
    sweep_path: str,
    values: list,
    extra_tags: tuple[str, ...] = (),
    description: str | None = None,
) -> list[ExperimentSpec]:
    """Register one variant per sweep value. Each variant gets a unique name
    `{name_prefix}_{value_str}`, the shared tags ("sweep", f"sweep:{sweep_path}",
    name_prefix, ...base.tags), and lives under results/<prefix>/<frag>/."""
    sweep_tags = tuple(dict.fromkeys(  # preserve order, dedup
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


def _fmt_value(v: Any) -> str:
    """Safe identifier fragment for a sweep value.
    0.95 -> "0p95", 1e-09 -> "1e-09", (3,3,8,12) -> "3x3x8x12"."""
    if isinstance(v, float):
        s = repr(v)
        if "e" in s or "E" in s:
            return s
        return s.replace(".", "p").replace("-", "m")
    if isinstance(v, (tuple, list)):
        return "x".join(_fmt_value(x) for x in v)
    return str(v)


def override_at_path(spec: ExperimentSpec, path: str, value: Any) -> ExperimentSpec:
    """Return a copy of `spec` with the dotted `path` replaced by `value`.
    Paths: top-level fields (gamma, n_episodes, eval_episodes, seeds),
    agent.hyperparams.<key>, or env.kwargs.<key>."""
    parts = path.split(".")

    if len(parts) == 3 and parts[0] == "agent" and parts[1] == "hyperparams":
        new_hp = {**spec.agent.hyperparams, parts[2]: value}
        return dataclasses.replace(
            spec, agent=dataclasses.replace(spec.agent, hyperparams=new_hp)
        )

    if len(parts) == 3 and parts[0] == "env" and parts[1] == "kwargs":
        new_kw = {**spec.env.kwargs, parts[2]: value}
        return dataclasses.replace(
            spec, env=dataclasses.replace(spec.env, kwargs=new_kw)
        )

    if len(parts) == 1 and parts[0] in {"gamma", "n_episodes", "eval_episodes", "seeds"}:
        return dataclasses.replace(spec, **{parts[0]: value})

    raise ValueError(
        f"Unsupported override path {path!r}. Use 'gamma', 'n_episodes', "
        f"'eval_episodes', 'seeds', 'agent.hyperparams.<key>', "
        f"or 'env.kwargs.<key>'."
    )


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
    """Sweep-variant override: apply `value` at `sweep_path` and overwrite
    identity metadata (name, tags, description, results_path_parts).
    """
    spec = override_at_path(base, sweep_path, value)
    return dataclasses.replace(
        spec,
        name=new_name,
        tags=new_tags,
        description=new_description,
        results_path_parts=new_results_path_parts,
    )


# ===========================================================================
# Registered experiments
# ===========================================================================

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

# --- Phase 2b: tabular HP sweeps -------------------------------------------
# Two HPs per tabular algorithm on Blackjack (α and ε-decay horizon), which
# is the FAQ minimum. The n_episodes sweep that used to live here was
# dropped: training budget is not a hyperparameter in the rubric sense,
# and the default runs' learning curves already illustrate convergence.

_ALPHA_VALUES = [0.01, 0.05, 0.1, 0.2]
_EPS_DECAY_VALUES = [10_000, 50_000, 100_000, 200_000]

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

# Default binning follows the Spring 2026 FAQ's starter grid: (3, 3, 8, 12).
# Angle and angular velocity get the finest resolution because they dominate
# the control problem; cart position / velocity are coarser. 3*3*8*12 = 864
# non-terminal bins. We sweep over this in the discretization study below.
_CARTPOLE_DEFAULT_N_BINS = (3, 3, 8, 12)

_CARTPOLE_ENV = EnvSpec(
    name="cartpole",
    kwargs={"n_bins": _CARTPOLE_DEFAULT_N_BINS},
)

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
         description="SARSA on discretized CartPole-v1 with (3,3,8,12) binning "
                     "(FAQ starter grid; 864 bins). CartPole-v1 'solved' = 475; "
                     "tabular aims for >= 150 mean return."))
register(dataclasses.replace(_CARTPOLE_QLEARNING_BASE, name="cartpole_qlearning_default",
         description="Q-learning on discretized CartPole-v1 with (3,3,8,12) "
                     "binning; off-policy analogue of cartpole_sarsa_default."))

# --- CartPole HP sweeps (tabular SARSA / Q-Learning) --------------------------
# Per the FAQ: >=2 validated HPs with 5 seeds per agent, plus an n_bins sweep
# for the "assess discretization" question. Three dims each: alpha, gamma, n_bins.

# alpha sweep -----------------------------------------------------------------
register_sweep(
    name_prefix="cartpole_sarsa_alpha_sweep",
    base=_CARTPOLE_SARSA_BASE,
    sweep_path="agent.hyperparams.alpha",
    values=[0.05, 0.1, 0.2, 0.5],
    extra_tags=("hp_sweep", "alpha"),
    description="SARSA on CartPole: step-size sweep. Smaller alpha => "
                "more stable but slower; larger alpha => faster but noisier.",
)
register_sweep(
    name_prefix="cartpole_qlearning_alpha_sweep",
    base=_CARTPOLE_QLEARNING_BASE,
    sweep_path="agent.hyperparams.alpha",
    values=[0.05, 0.1, 0.2, 0.5],
    extra_tags=("hp_sweep", "alpha"),
    description="Q-Learning on CartPole: step-size sweep (off-policy twin of "
                "cartpole_sarsa_alpha_sweep).",
)

# gamma sweep -----------------------------------------------------------------
register_sweep(
    name_prefix="cartpole_sarsa_gamma_sweep",
    base=_CARTPOLE_SARSA_BASE,
    sweep_path="gamma",
    values=[0.9, 0.95, 0.99, 1.0],
    extra_tags=("hp_sweep", "gamma"),
    description="SARSA on CartPole: discount sweep. CartPole's long horizon "
                "(up to 500 steps) makes gamma a first-class hyperparameter "
                "here, unlike Blackjack where gamma barely matters.",
)
register_sweep(
    name_prefix="cartpole_qlearning_gamma_sweep",
    base=_CARTPOLE_QLEARNING_BASE,
    sweep_path="gamma",
    values=[0.9, 0.95, 0.99, 1.0],
    extra_tags=("hp_sweep", "gamma"),
    description="Q-Learning on CartPole: discount sweep (off-policy twin of "
                "cartpole_sarsa_gamma_sweep).",
)

# n_bins sweep ----------------------------------------------------------------
# Grids span ~1.5 orders of magnitude in state count:
#   (1, 1, 6, 6)    =   36 states (cart state ignored entirely)
#   (3, 3, 6, 6)    =  324 states
#   (3, 3, 8, 12)   =  864 states  <- FAQ-recommended baseline
#   (5, 5, 12, 16)  = 4800 states  <- fine-grained
_CARTPOLE_NBIN_GRIDS = [
    (1, 1, 6, 6),
    (3, 3, 6, 6),
    (3, 3, 8, 12),
    (5, 5, 12, 16),
]
register_sweep(
    name_prefix="cartpole_sarsa_nbins_sweep",
    base=_CARTPOLE_SARSA_BASE,
    sweep_path="env.kwargs.n_bins",
    values=_CARTPOLE_NBIN_GRIDS,
    extra_tags=("hp_sweep", "discretization"),
    description="SARSA on CartPole: discretization sweep from very coarse "
                "(36 bins) to fine (4800 bins). Expected to show a bias-"
                "variance trade-off: coarser grids train fast but cap "
                "policy quality; finer grids need more samples to visit.",
)
register_sweep(
    name_prefix="cartpole_qlearning_nbins_sweep",
    base=_CARTPOLE_QLEARNING_BASE,
    sweep_path="env.kwargs.n_bins",
    values=_CARTPOLE_NBIN_GRIDS,
    extra_tags=("hp_sweep", "discretization"),
    description="Q-Learning on CartPole: discretization sweep (off-policy "
                "twin of cartpole_sarsa_nbins_sweep).",
)

# --- Phase 3b: DP on CartPole with an estimated MDP ---------------------------
# CartPole's ODE has no clean tabular transition model, so we estimate T̂, R̂
# from random rollouts (FAQ-sanctioned) and run VI/PI on that. Evaluation
# still uses the real DiscretizedCartPole.

_CARTPOLE_EST_ENV = EnvSpec(
    name="cartpole_estimated",
    kwargs={
        "n_bins": _CARTPOLE_DEFAULT_N_BINS,      # FAQ default (3,3,8,12)
        "n_sampling_episodes": 5_000,            # ~100k transitions with random policy
        "sampling_policy": "random",
        "max_steps_per_episode": 500,
        # sampling_seed=None in the class means "use the runner's per-seed
        # seed", so each of our 5 seeds estimates its own MDP from a
        # different random rollout stream. That gives genuine variance in
        # the DP-derived policy as a function of model-estimation noise.
    },
)

_CARTPOLE_VI_BASE = ExperimentSpec(
    name="cartpole_vi_base",
    env=_CARTPOLE_EST_ENV,
    agent=AgentSpec(name="vi", hyperparams={
        "theta": 1e-4,           # looser than Blackjack's 1e-6 because
                                 # gamma=0.99 + survival rewards make VI
                                 # slow to reach 1e-6
        "max_sweeps": 3000,      # headroom for (5,5,12,16)
    }),
    n_episodes=0,                # VI/PI don't train via episodes
    eval_episodes=100,
    seeds=(0, 1, 2, 3, 4),
    gamma=0.99,
    tags=("cartpole", "vi", "dp", "estimated_mdp"),
)

_CARTPOLE_PI_BASE = ExperimentSpec(
    name="cartpole_pi_base",
    env=_CARTPOLE_EST_ENV,
    agent=AgentSpec(name="pi", hyperparams={
        "theta": 1e-4,
        "eval_max_sweeps": 1000,
        "max_outer_iters": 50,
    }),
    n_episodes=0,
    eval_episodes=100,
    seeds=(0, 1, 2, 3, 4),
    gamma=0.99,
    tags=("cartpole", "pi", "dp", "estimated_mdp"),
)

register(dataclasses.replace(_CARTPOLE_VI_BASE, name="cartpole_vi_default",
         description="Value iteration on an empirically-estimated CartPole "
                     "MDP (5k random-policy rollouts). Evaluated on real "
                     "CartPole dynamics. FAQ-default (3,3,8,12) binning."))
register(dataclasses.replace(_CARTPOLE_PI_BASE, name="cartpole_pi_default",
         description="Policy iteration on the same empirically-estimated "
                     "CartPole MDP. Paired with cartpole_vi_default for a "
                     "direct VI-vs-PI convergence comparison."))

# Discretization sweep — the CartPole-specific DP study the rubric asks for.
register_sweep(
    name_prefix="cartpole_vi_nbins_sweep",
    base=_CARTPOLE_VI_BASE,
    sweep_path="env.kwargs.n_bins",
    values=_CARTPOLE_NBIN_GRIDS,
    extra_tags=("hp_sweep", "discretization"),
    description="VI on estimated CartPole MDP: discretization sweep. "
                "Coarser grids sample better (more visits per state) but "
                "alias distinct states; finer grids are more expressive "
                "but under-sampled. The sweet spot is empirical.",
)
register_sweep(
    name_prefix="cartpole_pi_nbins_sweep",
    base=_CARTPOLE_PI_BASE,
    sweep_path="env.kwargs.n_bins",
    values=_CARTPOLE_NBIN_GRIDS,
    extra_tags=("hp_sweep", "discretization"),
    description="PI twin of cartpole_vi_nbins_sweep. Also exposes the "
                "VI-vs-PI convergence-speed comparison across grid sizes.",
)

# Sampling-budget sweep — how much data does the model-based path need?
_CARTPOLE_SAMPLE_BUDGETS = [500, 2000, 5000, 10_000]
register_sweep(
    name_prefix="cartpole_vi_samples_sweep",
    base=_CARTPOLE_VI_BASE,
    sweep_path="env.kwargs.n_sampling_episodes",
    values=_CARTPOLE_SAMPLE_BUDGETS,
    extra_tags=("hp_sweep", "sample_complexity"),
    description="VI on estimated CartPole MDP: sampling-budget sweep. "
                "Shows how policy quality improves as the estimated T,R "
                "gets more accurate. Complements the tabular RL sample-"
                "complexity story from Blackjack.",
)

# --- Phase 3c: exploration-policy study for MDP estimation --------------------
# Random sampling under-covers the upright-pole manifold on fine grids
# (rollouts die in ~25 steps). ε-greedy on a trained SARSA policy shifts
# sampling density toward states a good controller actually visits. Pilot
# runs showed ε ∈ [0.3, 0.7] is the interesting region.

_CARTPOLE_EPS_GRID = [0.1, 0.3, 0.5, 0.7]

def _vi_trained_sampling_base(
    n_bins: tuple[int, int, int, int],
    source_experiment: str,
    name_suffix: str,
) -> ExperimentSpec:
    env = EnvSpec(
        name="cartpole_estimated",
        kwargs={
            "n_bins": n_bins,
            "n_sampling_episodes": 5_000,
            "sampling_policy": "epsilon_greedy",
            "sampling_source_experiment": source_experiment,
            "sampling_epsilon": 0.5,  # overwritten by the sweep
            "max_steps_per_episode": 500,
        },
    )
    return dataclasses.replace(_CARTPOLE_VI_BASE,
                               name=f"cartpole_vi_trained_eps_{name_suffix}",
                               env=env,
                               tags=_CARTPOLE_VI_BASE.tags + ("trained_sampling",))

_VI_TRAINED_3X3X8X12 = _vi_trained_sampling_base(
    n_bins=(3, 3, 8, 12),
    source_experiment="cartpole_sarsa_nbins_sweep_3x3x8x12",
    name_suffix="3x3x8x12",
)
_VI_TRAINED_5X5X12X16 = _vi_trained_sampling_base(
    n_bins=(5, 5, 12, 16),
    source_experiment="cartpole_sarsa_nbins_sweep_5x5x12x16",
    name_suffix="5x5x12x16",
)

register_sweep(
    name_prefix="cartpole_vi_trained_eps_3x3x8x12",
    base=_VI_TRAINED_3X3X8X12,
    sweep_path="env.kwargs.sampling_epsilon",
    values=_CARTPOLE_EPS_GRID,
    extra_tags=("hp_sweep", "exploration_policy"),
    description="VI on estimated CartPole MDP at (3,3,8,12), sampling "
                "transitions via ε-greedy on a trained SARSA policy. Sweep "
                "over ε tests how reliant the model-estimation is on the "
                "sampling policy quality vs. raw exploration.",
)
register_sweep(
    name_prefix="cartpole_vi_trained_eps_5x5x12x16",
    base=_VI_TRAINED_5X5X12X16,
    sweep_path="env.kwargs.sampling_epsilon",
    values=_CARTPOLE_EPS_GRID,
    extra_tags=("hp_sweep", "exploration_policy"),
    description="Same ε sweep at the finest grid (5,5,12,16), where random "
                "sampling already works surprisingly well. Tests whether "
                "trained-policy sampling is uniformly beneficial or "
                "grid-dependent.",
)


# --- Phase 4: DQN + Rainbow-medium ablation (EC) ------------------------------
# 6 configs: baseline + {Double, Dueling, PER, N-step} + full Rainbow.
# HPs are shared; only component toggles change (Hessel et al. 2018
# clean-sweep design). C51/NoisyNets are intentionally out of scope.

_DQN_SHARED_HP: dict[str, Any] = {
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

_DQN_BASE = ExperimentSpec(
    name="dqn_ablation_baseline",
    env=EnvSpec(name="cartpole_continuous"),
    agent=AgentSpec(name="dqn", hyperparams=dict(_DQN_SHARED_HP)),
    n_episodes=300,
    eval_episodes=20,
    seeds=(0, 1, 2, 3, 4),
    gamma=0.99,
    tags=("dqn", "rainbow_ablation", "baseline"),
    description="Vanilla DQN baseline on CartPole-v1 (continuous state, MLP Q-net). "
                "No Double / Dueling / PER / N-step. Reference bar for the ablation.",
    results_path_parts=("dqn_ablation", "baseline"),
)


def _dqn_variant(
    *,
    suffix: str,
    extra_hp: dict[str, Any],
    tag: str,
    description: str,
) -> None:
    """Register one Rainbow ablation variant. Shares _DQN_SHARED_HP."""
    hp = dict(_DQN_SHARED_HP)
    hp.update(extra_hp)
    register(dataclasses.replace(
        _DQN_BASE,
        name=f"dqn_ablation_{suffix}",
        agent=AgentSpec(name="dqn", hyperparams=hp),
        tags=("dqn", "rainbow_ablation", tag),
        description=description,
        results_path_parts=("dqn_ablation", suffix),
    ))


register(_DQN_BASE)

_dqn_variant(
    suffix="double",
    extra_hp={"double": True},
    tag="double",
    description="DQN + Double-DQN: decouples action-selection (online net) "
                "from value-estimation (target net). Expected to reduce the "
                "positive bias of max-over-target that vanilla DQN suffers from.",
)

_dqn_variant(
    suffix="dueling",
    extra_hp={"dueling": True},
    tag="dueling",
    description="DQN + Dueling network: V(s) + A(s,·) with mean-centered "
                "advantages. Mostly architectural — pays off more in "
                "environments where many actions have similar values.",
)

_dqn_variant(
    suffix="per",
    extra_hp={"per": True},
    tag="per",
    description="DQN + Prioritized Experience Replay (proportional, sum-tree). "
                "Samples transitions with high TD-error more often; β anneals "
                "from 0.4 to 1.0 over 20k gradient steps to correct IS bias.",
)

_dqn_variant(
    suffix="nstep",
    extra_hp={"nstep": 3},
    tag="nstep",
    description="DQN + 3-step TD targets. Trades a bit of bias (off-policy "
                "error from using the behaviour policy's multi-step returns) "
                "for lower variance and faster credit assignment.",
)

_dqn_variant(
    suffix="rainbow",
    extra_hp={"double": True, "dueling": True, "per": True, "nstep": 3},
    tag="rainbow_full",
    description="Rainbow-medium: Double + Dueling + PER + 3-step. The all-in "
                "variant. Comparing this against the four single-component "
                "variants isolates each component's marginal contribution.",
)
