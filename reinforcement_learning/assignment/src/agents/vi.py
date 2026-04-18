"""Value Iteration.

Classic synchronous Bellman-optimality update, in-place (Gauss-Seidel). Solves

    V*(s) = max_a  sum_{s',r} T(s,a,s',r) [ r + gamma V*(s') ]

by repeated fixed-point application. Contraction with factor gamma under the
sup-norm, so convergence is guaranteed for gamma < 1 (Sutton & Barto §4.4).

The env passed to `run` must implement the **MDP-model** side of the env
contract:

- `all_states()` → iterable of hashable states
- `is_terminal(state)` → bool
- `transitions(state, action)` → list[(prob, next_state, reward)]
- `N_ACTIONS` → int (we assume actions are 0..N_ACTIONS-1)

plus the **rollout** side (`reset()`, `step(a)`) so we can evaluate the
greedy policy by simulation after solving.

History recorded per sweep:
  - sweep_deltas: max |V_new(s) - V_old(s)| over all states (Bellman residual)
  - sweep_wall_times: wall-clock seconds for the sweep
  - greedy_policy_changes: how many states changed their greedy action this
    sweep (value convergence usually runs ahead of policy convergence, and
    plotting both is a standard DP diagnostic)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.agents.base import BaseAgent, RunResult


class ValueIteration(BaseAgent):
    """Synchronous in-place value iteration."""

    name = "vi"

    def __init__(
        self,
        theta: float = 1e-6,
        max_sweeps: int = 1000,
    ):
        """
        Args:
            theta: Bellman-residual threshold for early stopping.
            max_sweeps: hard cap on sweeps — prevents runaway loops if the
                MDP is pathological or gamma is too close to 1.
        """
        self.theta = theta
        self.max_sweeps = max_sweeps

    def run(
        self,
        env,
        *,
        n_episodes: int,  # ignored; DP doesn't train via episodes
        eval_episodes: int,
        gamma: float,
        seed: int,
    ) -> RunResult:
        _require_mdp_interface(env)
        t0 = time.perf_counter()

        states = list(env.all_states())
        V: dict[Any, float] = {s: 0.0 for s in states}
        # Default action 0 for all states. Terminal-state entries are never
        # updated (terminals have no meaningful action) but we keep a
        # consistent default to match PI's initial policy and make output
        # comparisons clean.
        prev_greedy: dict[Any, int] = {s: 0 for s in states}

        sweep_deltas: list[float] = []
        sweep_times: list[float] = []
        policy_change_counts: list[int] = []

        for sweep in range(self.max_sweeps):
            sweep_t0 = time.perf_counter()
            delta = 0.0
            policy_changes = 0

            for s in states:
                if env.is_terminal(s):
                    V[s] = 0.0
                    continue
                v_old = V[s]
                best_value, best_action = _best_action_value(env, s, V, gamma)
                V[s] = best_value
                delta = max(delta, abs(v_old - best_value))
                if best_action != prev_greedy[s]:
                    policy_changes += 1
                    prev_greedy[s] = best_action

            sweep_deltas.append(delta)
            sweep_times.append(time.perf_counter() - sweep_t0)
            policy_change_counts.append(policy_changes)

            if delta < self.theta:
                break

        policy = dict(prev_greedy)
        Q = _compute_Q(env, V, gamma)
        eval_returns, eval_steps = _rollout_policy(
            env, policy, eval_episodes, seed=seed
        )

        history = {
            "sweep_deltas": sweep_deltas,
            "sweep_wall_times": sweep_times,
            "policy_change_counts": policy_change_counts,
            "converged_in_sweeps": len(sweep_deltas),
            "converged": sweep_deltas[-1] < self.theta,
            "final_delta": sweep_deltas[-1],
        }

        return RunResult(
            train_returns=[],
            train_steps=[],
            eval_returns=eval_returns,
            eval_steps=eval_steps,
            history=history,
            policy=policy,
            Q=Q,
            wall_clock_seconds=time.perf_counter() - t0,
        )


# ---------------------------------------------------------------------------
# Shared DP utilities (also used by PolicyIteration)
# ---------------------------------------------------------------------------

def _require_mdp_interface(env) -> None:
    """Fail fast with a clear error if the env isn't a full MDP model."""
    for method in ("all_states", "transitions", "is_terminal"):
        if not hasattr(env, method):
            raise TypeError(
                f"DP agents require the env to expose `{method}` "
                f"(MDP-model interface). Env {type(env).__name__} does not."
            )
    if not hasattr(env, "N_ACTIONS"):
        raise TypeError(
            f"Env {type(env).__name__} must expose `N_ACTIONS`."
        )


def _best_action_value(env, s, V, gamma: float) -> tuple[float, int]:
    """Compute max_a Q(s,a) and argmax_a Q(s,a)."""
    best_value = -float("inf")
    best_action = 0
    for a in range(env.N_ACTIONS):
        q = 0.0
        for p, s_next, r in env.transitions(s, a):
            q += p * (r + gamma * V[s_next])
        if q > best_value:
            best_value = q
            best_action = a
    return best_value, best_action


def _compute_Q(env, V: dict, gamma: float) -> dict:
    """Extract Q from V after solving. Nice for analysis / plotting."""
    Q: dict = {}
    for s in V:
        Q[s] = [0.0] * env.N_ACTIONS
        if env.is_terminal(s):
            continue
        for a in range(env.N_ACTIONS):
            q = 0.0
            for p, s_next, r in env.transitions(s, a):
                q += p * (r + gamma * V[s_next])
            Q[s][a] = q
    return Q


def _rollout_policy(
    env,
    policy: dict,
    n_rollouts: int,
    seed: int,
    max_steps: int = 1000,
) -> tuple[list[float], list[int]]:
    """Run `n_rollouts` greedy-policy episodes, returning (returns, steps)."""
    rng = np.random.default_rng(seed + 10_000)
    returns = []
    steps = []
    for ep in range(n_rollouts):
        # Some envs accept a per-reset seed; pass a derived one for determinism.
        try:
            s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        except TypeError:
            s = env.reset()
        total = 0.0
        for t in range(max_steps):
            a = policy.get(s, 0)
            s, r, done, _ = env.step(a)
            total += r
            if done:
                steps.append(t + 1)
                break
        else:
            steps.append(max_steps)
        returns.append(total)
    return returns, steps
