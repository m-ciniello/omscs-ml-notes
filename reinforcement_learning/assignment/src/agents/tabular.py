"""Shared building blocks for tabular model-free agents.

SARSA and Q-Learning differ only in the TD target; everything else (Q-table,
ε-greedy, linear ε decay, greedy eval) lives here. Dict-backed Q-table with
lazy zero init avoids encoding tuple states to integer indices.
"""

from __future__ import annotations

import numpy as np


class QTable:
    """Dict-backed Q-table. `values(s)` returns the mutable zero-inited row."""

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)
        self._table: dict = {}

    def values(self, state) -> np.ndarray:
        q = self._table.get(state)
        if q is None:
            q = np.zeros(self.n_actions, dtype=float)
            self._table[state] = q
        return q

    def greedy_action(self, state) -> int:
        return int(np.argmax(self.values(state)))

    def to_dict(self) -> dict:
        return {s: q.tolist() for s, q in self._table.items()}

    def policy_dict(self) -> dict:
        return {s: int(np.argmax(q)) for s, q in self._table.items()}

    def n_visited_states(self) -> int:
        return len(self._table)


def epsilon_greedy(
    qtable: QTable,
    state,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(qtable.n_actions))
    return qtable.greedy_action(state)


def linear_epsilon(
    episode: int,
    *,
    start: float,
    end: float,
    decay_episodes: int,
) -> float:
    """Linear from `start` to `end` over the first `decay_episodes`, then
    clamped at `end`. decay_episodes <= 0 returns `end` immediately."""
    if decay_episodes <= 0:
        return end
    t = min(episode, decay_episodes) / decay_episodes
    return start + (end - start) * t


def eval_greedy_policy(
    env,
    qtable: QTable,
    *,
    eval_episodes: int,
    max_steps_per_episode: int,
) -> tuple[list[float], list[int]]:
    """Roll out the greedy policy; return (returns, episode-step-counts)."""
    returns: list[float] = []
    steps: list[int] = []
    for _ in range(eval_episodes):
        state = env.reset()
        total = 0.0
        ep_steps = max_steps_per_episode
        for t in range(max_steps_per_episode):
            action = qtable.greedy_action(state)
            state, reward, done, _ = env.step(action)
            total += reward
            if done:
                ep_steps = t + 1
                break
        steps.append(ep_steps)
        returns.append(total)
    return returns, steps
