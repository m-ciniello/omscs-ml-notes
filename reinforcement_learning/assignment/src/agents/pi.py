"""Policy Iteration.

Alternates between two steps (Sutton & Barto §4.3):

  1. Policy evaluation: solve V^pi(s) = sum_{s',r} T(s, pi(s), s', r) [r + gamma V^pi(s')]
     by iterative sweeps under the *fixed* current policy. This is a smaller
     fixed point (linear in V, no max over actions).
  2. Policy improvement: set pi(s) = argmax_a sum_{s',r} T(s,a,s',r) [r + gamma V^pi(s')].

Terminates when the greedy policy stops changing (policy stable). Because the
policy space is finite, this strictly improves until a fixed point — so PI
converges in a finite number of *outer* iterations, typically far fewer than
VI's sweep count. The cost is that each outer iteration solves a full linear
system (approximately).

Tracked history:
  - outer_iters: number of (eval + improve) rounds completed
  - eval_sweeps_per_outer: how many sweeps policy evaluation took each round
  - policy_changes_per_outer: number of states whose greedy action flipped
  - bellman_residual_per_outer: final delta at the end of each eval phase
  - wall_times_per_outer: wall-clock per outer iteration

These mirror the VI history format so side-by-side plots are easy.
"""

from __future__ import annotations

import time
from typing import Any

from src.agents.base import BaseAgent, RunResult
from src.agents.vi import (
    _best_action_value,
    _compute_Q,
    _require_mdp_interface,
    _rollout_policy,
)


class PolicyIteration(BaseAgent):
    """Policy iteration with iterative policy evaluation."""

    name = "pi"

    def __init__(
        self,
        theta: float = 1e-6,
        eval_max_sweeps: int = 1000,
        max_outer_iters: int = 100,
    ):
        """
        Args:
            theta: Bellman-residual threshold for policy evaluation.
            eval_max_sweeps: hard cap on sweeps per evaluation phase.
            max_outer_iters: hard cap on outer (eval+improve) iterations.
        """
        self.theta = theta
        self.eval_max_sweeps = eval_max_sweeps
        self.max_outer_iters = max_outer_iters

    def run(
        self,
        env,
        *,
        n_episodes: int,  # ignored
        eval_episodes: int,
        gamma: float,
        seed: int,
    ) -> RunResult:
        _require_mdp_interface(env)
        t0 = time.perf_counter()

        states = list(env.all_states())
        V: dict[Any, float] = {s: 0.0 for s in states}
        # Initial policy: all zeros. Arbitrary — PI converges regardless.
        policy: dict[Any, int] = {s: 0 for s in states}

        eval_sweeps_per_outer: list[int] = []
        policy_changes_per_outer: list[int] = []
        bellman_residual_per_outer: list[float] = []
        wall_times_per_outer: list[float] = []

        outer_iter = 0
        for outer_iter in range(1, self.max_outer_iters + 1):
            iter_t0 = time.perf_counter()

            sweeps, final_delta = self._policy_evaluation(env, policy, V, gamma)
            eval_sweeps_per_outer.append(sweeps)
            bellman_residual_per_outer.append(final_delta)

            policy_changes = self._policy_improvement(env, policy, V, gamma)
            policy_changes_per_outer.append(policy_changes)

            wall_times_per_outer.append(time.perf_counter() - iter_t0)

            if policy_changes == 0:
                break

        Q = _compute_Q(env, V, gamma)
        eval_returns, eval_steps = _rollout_policy(
            env, policy, eval_episodes, seed=seed
        )

        history = {
            "outer_iters": outer_iter,
            "eval_sweeps_per_outer": eval_sweeps_per_outer,
            "policy_changes_per_outer": policy_changes_per_outer,
            "bellman_residual_per_outer": bellman_residual_per_outer,
            "wall_times_per_outer": wall_times_per_outer,
            "total_eval_sweeps": sum(eval_sweeps_per_outer),
            "converged": (policy_changes_per_outer[-1] == 0),
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

    # ---- inner helpers ----

    def _policy_evaluation(self, env, policy, V, gamma) -> tuple[int, float]:
        """In-place iterative policy evaluation. Mutates V; returns (sweeps, delta).

        Solves V^pi by repeatedly applying the *policy-fixed* Bellman operator
        until the per-state change falls below theta. Much cheaper than VI's
        operator because there's no max over actions — just a single weighted
        sum per state.
        """
        for sweep in range(1, self.eval_max_sweeps + 1):
            delta = 0.0
            for s in list(V.keys()):
                if env.is_terminal(s):
                    V[s] = 0.0
                    continue
                v_old = V[s]
                a = policy[s]
                new_v = 0.0
                for p, s_next, r in env.transitions(s, a):
                    new_v += p * (r + gamma * V[s_next])
                V[s] = new_v
                delta = max(delta, abs(v_old - new_v))
            if delta < self.theta:
                return sweep, delta
        return self.eval_max_sweeps, delta

    def _policy_improvement(self, env, policy, V, gamma) -> int:
        """Greedy policy improvement. Mutates policy; returns # of changes."""
        changes = 0
        for s in list(policy.keys()):
            if env.is_terminal(s):
                continue
            _, best_action = _best_action_value(env, s, V, gamma)
            if best_action != policy[s]:
                policy[s] = best_action
                changes += 1
        return changes
