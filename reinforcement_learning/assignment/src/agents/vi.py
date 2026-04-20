"""Value Iteration: in-place (Gauss-Seidel) Bellman-optimality sweeps.

Env must expose the MDP-model side (`all_states`, `transitions`,
`is_terminal`, `N_ACTIONS`) for the sweeps, and the rollout side
(`reset`, `step`) for greedy-policy evaluation after solving.

Per-sweep history (for the convergence plots):
- `sweep_deltas`: Bellman residual (max |ΔV|)
- `sweep_wall_times`: seconds per sweep
- `policy_change_counts`: #states whose greedy action flipped this sweep
  (value convergence runs ahead of policy convergence — standard diagnostic)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class ValueIteration:
    name = "vi"

    def __init__(
        self,
        theta: float = 1e-6,
        max_sweeps: int = 1000,
    ):
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
    ) -> dict:
        t0 = time.perf_counter()

        states = list(env.all_states())
        V: dict[Any, float] = {s: 0.0 for s in states}
        # default action 0 everywhere (incl. terminals) so PI's init matches
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

        return {
            "train_returns": [],
            "train_steps": [],
            "eval_returns": eval_returns,
            "eval_steps": eval_steps,
            "history": history,
            "policy": policy,
            "Q": Q,
            "wall_clock_seconds": time.perf_counter() - t0,
        }


# --- Shared DP utilities (also used by PolicyIteration) ---

def _q_values(env, s, V: dict, gamma: float) -> list[float]:
    """Bellman backup: Q(s, a) = Σ_{s'} T(s'|s,a) [R + γ V(s')] for each a."""
    q = [0.0] * env.N_ACTIONS
    for a in range(env.N_ACTIONS):
        for p, s_next, r in env.transitions(s, a):
            q[a] += p * (r + gamma * V[s_next])
    return q


def _best_action_value(env, s, V, gamma: float) -> tuple[float, int]:
    """(max_a Q(s,a), argmax_a Q(s,a))."""
    q = _q_values(env, s, V, gamma)
    best_action = max(range(len(q)), key=q.__getitem__)
    return q[best_action], best_action


def _compute_Q(env, V: dict, gamma: float) -> dict:
    """Extract Q from V after solving."""
    return {
        s: [0.0] * env.N_ACTIONS if env.is_terminal(s)
           else _q_values(env, s, V, gamma)
        for s in V
    }


def _rollout_policy(
    env,
    policy: dict,
    n_rollouts: int,
    seed: int,
    max_steps: int = 1000,
) -> tuple[list[float], list[int]]:
    """Run `n_rollouts` greedy-policy episodes; returns (returns, steps)."""
    rng = np.random.default_rng(seed + 10_000)
    returns = []
    steps = []
    for ep in range(n_rollouts):
        # pass a derived per-reset seed when the env supports it (determinism)
        try:
            s = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        except TypeError:
            s = env.reset()
        total = 0.0
        ep_steps = max_steps
        for t in range(max_steps):
            a = policy.get(s, 0)
            s, r, done, _ = env.step(a)
            total += r
            if done:
                ep_steps = t + 1
                break
        steps.append(ep_steps)
        returns.append(total)
    return returns, steps
