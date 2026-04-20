"""Phase 3b — VI / PI on an empirically-estimated CartPole MDP.

CartPole's ODE has no natural tabular model, so we estimate T̂, R̂ from
rollouts (FAQ-sanctioned) and run VI / PI on the estimate. Evaluation
still uses the real dynamics, not the estimate.

Three studies:
  1. nbins sweep — grid resolution vs policy quality at fixed sample budget.
  2. sample budget sweep — how much data does the estimate need?
  3. trained-ε sweep (H5) — replace random sampling with ε-greedy on top
     of a trained SARSA Q-table. Tests the hypothesis that sampling-policy
     coverage, not DP, is the bottleneck on CartPole.
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
from src.configs.cartpole_tabular import (
    CARTPOLE_DEFAULT_N_BINS,
    CARTPOLE_NBIN_GRIDS,
)


_EST_ENV = EnvSpec(
    name="cartpole_estimated",
    kwargs={
        "n_bins": CARTPOLE_DEFAULT_N_BINS,
        "n_sampling_episodes": 5_000,          # ~100k transitions with random policy
        "sampling_policy": "random",
        "max_steps_per_episode": 500,
        # sampling_seed=None => use the runner's per-seed seed, so each of
        # the 10 seeds estimates its own MDP from a distinct rollout stream
        # (genuine model-estimation variance in the DP-derived policy).
    },
)

_VI_BASE = ExperimentSpec(
    name="cartpole_vi_base",  # template; NOT registered directly
    env=_EST_ENV,
    agent=AgentSpec(name="vi", hyperparams={
        "theta": 1e-4,           # looser than Blackjack's 1e-9 because
                                 # γ=0.99 + per-step rewards make VI slow
                                 # to reach machine precision
        "max_sweeps": 3000,      # headroom for (5,5,12,16)
    }),
    n_episodes=0,                # DP doesn't train via episodes
    eval_episodes=100,
    seeds=tuple(range(10)),
    gamma=0.99,
)

_PI_BASE = ExperimentSpec(
    name="cartpole_pi_base",  # template; NOT registered directly
    env=_EST_ENV,
    agent=AgentSpec(name="pi", hyperparams={
        "theta": 1e-4,
        "eval_max_sweeps": 1000,
        "max_outer_iters": 50,
    }),
    n_episodes=0,
    eval_episodes=100,
    seeds=tuple(range(10)),
    gamma=0.99,
)

# --- Reference single-point runs ------------------------------------------
register(dataclasses.replace(
    _VI_BASE, name="cartpole_vi_default",
    description="VI on an empirically-estimated CartPole MDP "
                "(5k random-policy rollouts, (3,3,8,12) binning). "
                "Evaluated on real CartPole dynamics.",
))
register(dataclasses.replace(
    _PI_BASE, name="cartpole_pi_default",
    description="PI on the same empirically-estimated CartPole MDP. "
                "Paired with cartpole_vi_default for VI-vs-PI comparison.",
))

# --- Discretization sweep (CartPole-specific DP study) --------------------
register_sweep(
    name_prefix="cartpole_vi_nbins_sweep",
    base=_VI_BASE,
    sweep_path="env.kwargs.n_bins",
    values=CARTPOLE_NBIN_GRIDS,
    description="VI on estimated CartPole MDP: discretization sweep. "
                "Coarse grids sample better (more visits per state) but "
                "alias distinct states; fine grids are more expressive "
                "but under-sampled. Sweet spot is empirical.",
)
register_sweep(
    name_prefix="cartpole_pi_nbins_sweep",
    base=_PI_BASE,
    sweep_path="env.kwargs.n_bins",
    values=CARTPOLE_NBIN_GRIDS,
    description="PI twin of cartpole_vi_nbins_sweep. Also exposes the "
                "VI-vs-PI convergence-speed comparison across grid sizes.",
)

# --- Sampling-budget sweep -----------------------------------------------
# {500, 5000, 10000}: 2000 dropped as a redundant near-linear interior point.
_SAMPLE_BUDGETS = [500, 5000, 10_000]
register_sweep(
    name_prefix="cartpole_vi_samples_sweep",
    base=_VI_BASE,
    sweep_path="env.kwargs.n_sampling_episodes",
    values=_SAMPLE_BUDGETS,
    description="VI on estimated CartPole MDP: sampling-budget sweep. "
                "Shows how DP-derived policy quality improves as T̂, R̂ "
                "get more accurate.",
)

# --- Trained-policy sampling study (H5) ----------------------------------
# Random sampling under-covers the upright-pole manifold on fine grids
# (rollouts die in ~25 steps). ε-greedy on a trained SARSA policy shifts
# sampling density toward states a competent controller visits. Pilot
# runs showed ε ∈ [0.3, 0.7] is the interesting region.
_EPS_GRID = [0.1, 0.3, 0.5, 0.7]


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
    return dataclasses.replace(
        _VI_BASE,
        name=f"cartpole_vi_trained_eps_{name_suffix}",
        env=env,
    )


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
    values=_EPS_GRID,
    description="VI on estimated CartPole MDP at (3,3,8,12), sampling via "
                "ε-greedy on a trained SARSA policy. ε sweep tests how "
                "much model-estimation depends on sampling-policy "
                "quality vs raw exploration.",
)
register_sweep(
    name_prefix="cartpole_vi_trained_eps_5x5x12x16",
    base=_VI_TRAINED_5X5X12X16,
    sweep_path="env.kwargs.sampling_epsilon",
    values=_EPS_GRID,
    description="Same ε sweep at the finest grid (5,5,12,16). Tests "
                "whether trained-policy sampling is uniformly beneficial "
                "or grid-dependent.",
)
