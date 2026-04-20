"""Phase 1 — Value / Policy Iteration on the analytical Blackjack MDP.

Two reference runs (one per algorithm) at γ=1.0 and tight θ, plus two
1-D HP sweeps. The sweeps study how γ (contraction rate) and θ (stopping
tolerance) affect convergence *count*; on Blackjack neither is expected
to change the optimal policy (rewards are purely terminal, so γ = 1 is
honest and any γ ∈ [0.5, 1.0] recovers the same greedy policy).

Why γ=1 is the default here: Blackjack is strictly episodic, rewards are
only paid at termination (±1), and every policy is proper (episodes
terminate almost surely because `hit` strictly increases the player's
sum). γ < 1 would artificially penalise longer hands — an artefact of
discounting, not a property of the problem. Matches Sutton & Barto
Ch. 5 Example 5.3.
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


# Every Blackjack-DP experiment uses 10 seeds; per-run cost is ~1 s so the
# cost of doubling over the typical 5-seed campaign is negligible.
_BLACKJACK_VI_BASE = ExperimentSpec(
    name="blackjack_vi_base",  # template; NOT registered directly
    env=EnvSpec(name="blackjack"),
    agent=AgentSpec(name="vi", hyperparams={"theta": 1e-9, "max_sweeps": 500}),
    n_episodes=0,
    eval_episodes=20_000,
    seeds=tuple(range(10)),
    gamma=1.0,
)

_BLACKJACK_PI_BASE = ExperimentSpec(
    name="blackjack_pi_base",  # template; NOT registered directly
    env=EnvSpec(name="blackjack"),
    agent=AgentSpec(name="pi", hyperparams={
        "theta": 1e-9,
        "eval_max_sweeps": 500,
        "max_outer_iters": 50,
    }),
    n_episodes=0,
    eval_episodes=20_000,
    seeds=tuple(range(10)),
    gamma=1.0,
)

# --- Reference single-point runs (tight θ, γ=1.0) -------------------------
register(dataclasses.replace(
    _BLACKJACK_VI_BASE, name="blackjack_vi_default",
    description="VI on the analytical Blackjack MDP (γ=1.0, θ=1e-9), "
                "evaluated via 20k Gym rollouts. Reference point for "
                "the γ and θ sweeps below.",
))
register(dataclasses.replace(
    _BLACKJACK_PI_BASE, name="blackjack_pi_default",
    description="PI on the analytical Blackjack MDP (γ=1.0, θ=1e-9). "
                "Pair with blackjack_vi_default for VI-vs-PI convergence "
                "comparison.",
))

# --- HP sweeps -------------------------------------------------------------
# γ sweep: fix θ at the reference (1e-9) and walk γ down to 0.5. Shows
# that the policy is γ-invariant (rewards are terminal) while sweep-count
# drops as γ shrinks (stronger contraction). Keeping γ=0.95 adjacent to
# γ=1.0 is the cleanest way to see PI's PE-sweep count balloon at γ=1
# (no contraction in the PE inner loop).
_GAMMA_VALUES = [0.5, 0.8, 0.9, 0.95, 1.0]

register_sweep(
    name_prefix="blackjack_vi_gamma_sweep",
    base=_BLACKJACK_VI_BASE,
    sweep_path="gamma",
    values=_GAMMA_VALUES,
    description="VI on Blackjack: discount sweep at θ=1e-9. Tests the "
                "'γ-invariance of optimal policy' claim and exposes how "
                "γ affects VI sweep count via the log(1/θ)/log(1/γ) bound.",
)
register_sweep(
    name_prefix="blackjack_pi_gamma_sweep",
    base=_BLACKJACK_PI_BASE,
    sweep_path="gamma",
    values=_GAMMA_VALUES,
    description="PI on Blackjack: discount sweep at θ=1e-9. PI's outer-"
                "iter count is expected to stay at 2-3 regardless of γ; "
                "total policy-evaluation sweep count jumps at γ=1 where "
                "the PE contraction disappears.",
)

# θ sweep: fix γ at the reference (1.0) and walk θ from 1e-1 (coarse) to
# 1e-9 (tight). Shows the log(1/θ) growth in VI's sweep count and the
# near-invariance of PI's outer iterations.
_THETA_VALUES = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]

register_sweep(
    name_prefix="blackjack_vi_theta_sweep",
    base=_BLACKJACK_VI_BASE,
    sweep_path="agent.hyperparams.theta",
    values=_THETA_VALUES,
    description="VI on Blackjack: stopping-tolerance sweep at γ=1.0. "
                "Each ×10⁻² drop in θ adds ~2 sweeps (log(1/θ) scaling).",
)
register_sweep(
    name_prefix="blackjack_pi_theta_sweep",
    base=_BLACKJACK_PI_BASE,
    sweep_path="agent.hyperparams.theta",
    values=_THETA_VALUES,
    description="PI on Blackjack: PE-tolerance sweep at γ=1.0. Outer "
                "iterations should be ~flat across θ (Howard's result: "
                "policy convergence is combinatorial, independent of θ); "
                "total PE-sweep count rides θ directly.",
)
