"""Uniform-random action agent.

Serves two purposes:
1. Phase 0 smoke test: exercises the full runner + aggregator pipeline
   end-to-end before any real algorithms are wired up.
2. Baseline: sanity-check that learning agents actually learn something
   better than random on every problem we try.
"""

from __future__ import annotations

import time

import numpy as np

from src.agents.base import BaseAgent, RunResult


class RandomAgent(BaseAgent):
    """Picks each action uniformly at random. No learning."""

    name = "random"

    def __init__(self, max_steps_per_episode: int = 500):
        self.max_steps_per_episode = max_steps_per_episode

    def run(
        self,
        env,
        *,
        n_episodes: int,
        eval_episodes: int,
        gamma: float,
        seed: int,
    ) -> RunResult:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        train_returns = []
        train_steps = []
        for _ in range(n_episodes):
            ret, steps = self._rollout(env, rng)
            train_returns.append(ret)
            train_steps.append(steps)

        eval_returns = []
        eval_steps = []
        for _ in range(eval_episodes):
            ret, steps = self._rollout(env, rng)
            eval_returns.append(ret)
            eval_steps.append(steps)

        return RunResult(
            train_returns=train_returns,
            train_steps=train_steps,
            eval_returns=eval_returns,
            eval_steps=eval_steps,
            history={},
            policy=None,
            Q=None,
            wall_clock_seconds=time.perf_counter() - t0,
        )

    def _rollout(self, env, rng) -> tuple[float, int]:
        env.reset()
        total_return = 0.0
        for t in range(self.max_steps_per_episode):
            action = int(rng.integers(0, env.N_ACTIONS))
            _, reward, done, _ = env.step(action)
            total_return += reward
            if done:
                return total_return, t + 1
        return total_return, self.max_steps_per_episode
