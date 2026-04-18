"""Shared building blocks for tabular model-free agents.

SARSA and Q-Learning differ only in how they compute the TD target. Everything
else — the Q-table data structure, ε-greedy action selection, ε decay, greedy
evaluation — is identical. Those bits live here so the agent modules stay
focused on their update rule.

Design notes:

- **Dict-backed Q-table.** Blackjack and Gridworld both use tuple states, and
  Blackjack's state space is small (~360 states). A dict avoids having to
  encode states into integer indices and degrades gracefully as we add more
  tuple-state environments. Lookups are O(1) with hash overhead, which is
  dwarfed by Gymnasium step cost.

- **Lazy state initialisation.** `QTable.values(s)` allocates a zero row on
  first touch. This matches "Q(s,a) = 0 for all unseen (s,a)" semantics and
  means we only carry states we actually visited in the pickled result.

- **Linear ε schedule** covering the first `decay_episodes` episodes, then
  flat at `end`. Simple, monotone, trivial to sweep over.

- **Greedy eval is deterministic** (argmax with stable tie-breaking). Random
  tie-breaking would add noise to the metric without any learning benefit.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Q-table
# ---------------------------------------------------------------------------

class QTable:
    """Dict-backed Q-table with lazy zero initialisation.

    `values(s)` returns the *mutable* row for state s so update rules can
    write into it in place. This is the hottest path in training, so we avoid
    any defensive copying.
    """

    def __init__(self, n_actions: int):
        self.n_actions = int(n_actions)
        self._table: dict = {}

    def values(self, state) -> np.ndarray:
        """Return the (possibly newly-allocated) action-value row for `state`."""
        q = self._table.get(state)
        if q is None:
            q = np.zeros(self.n_actions, dtype=float)
            self._table[state] = q
        return q

    def greedy_action(self, state) -> int:
        """argmax over actions with numpy's stable tie-break (first maximum)."""
        return int(np.argmax(self.values(state)))

    def to_dict(self) -> dict:
        """Plain Python dict for pickling (numpy rows -> list[float])."""
        return {s: q.tolist() for s, q in self._table.items()}

    def policy_dict(self) -> dict:
        """state -> greedy action for every state we've ever touched."""
        return {s: int(np.argmax(q)) for s, q in self._table.items()}

    def n_visited_states(self) -> int:
        return len(self._table)


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def epsilon_greedy(
    qtable: QTable,
    state,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Standard ε-greedy: explore with prob ε, else take greedy action."""
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
    """Linear schedule from `start` to `end` over the first `decay_episodes`.

    After `decay_episodes`, clamps at `end`. If `decay_episodes <= 0`, returns
    `end` immediately — useful for ablations that want a constant ε.
    """
    if decay_episodes <= 0:
        return end
    t = min(episode, decay_episodes) / decay_episodes
    return start + (end - start) * t


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_greedy_policy(
    env,
    qtable: QTable,
    *,
    eval_episodes: int,
    max_steps_per_episode: int,
) -> tuple[list[float], list[int]]:
    """Roll out the greedy policy `eval_episodes` times, return (returns, steps).

    The env's rollout RNG is the single source of stochasticity in eval —
    we never explore here. Matches `RandomAgent._rollout` conventions.
    """
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
