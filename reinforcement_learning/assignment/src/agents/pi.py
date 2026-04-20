"""Policy Iteration: alternate (iterative eval, greedy improvement) until the
policy is stable. Far fewer outer iterations than VI sweeps, but each outer
iter solves a policy-fixed linear system. History mirrors VI's format so
side-by-side convergence plots are easy."""

from __future__ import annotations

import time
from typing import Any

from src.agents.vi import (
    _best_action_value,
    _compute_Q,
    _rollout_policy,
)


class PolicyIteration:
    name = "pi"

    def __init__(
        self,
        theta: float = 1e-6,
        eval_max_sweeps: int = 1000,
        max_outer_iters: int = 100,
    ):
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
    ) -> dict:
        t0 = time.perf_counter()

        states = list(env.all_states())
        V: dict[Any, float] = {s: 0.0 for s in states}
        policy: dict[Any, int] = {s: 0 for s in states}  # arbitrary init

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

    def _policy_evaluation(self, env, policy, V, gamma) -> tuple[int, float]:
        """In-place sweeps under a fixed policy. Mutates V; returns (sweeps, delta)."""
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
