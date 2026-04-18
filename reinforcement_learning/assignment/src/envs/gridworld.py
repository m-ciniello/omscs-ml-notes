"""Minimal deterministic gridworld used for Phase 0 smoke tests.

Start at (0, 0), goal at (rows-1, cols-1). Four actions: up/down/left/right.
Every step incurs -1 reward; reaching the goal terminates the episode with
reward 0. This is the simplest possible MDP that exercises the full runner
pipeline (episode termination, rewards, discrete state/action spaces, optional
transition model for DP).

Intentionally kept small (~80 lines) — this env is a smoke-test fixture,
not a research artefact. Real environments (Blackjack, CartPole) live in
dedicated modules and will supersede this for the actual assignment work.
"""

from __future__ import annotations


class GridWorld:
    """Deterministic rows x cols gridworld with a single absorbing goal."""

    # Action indices
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    N_ACTIONS = 4
    ACTION_NAMES = ("up", "down", "left", "right")
    _ACTION_DELTAS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }

    def __init__(self, rows: int = 3, cols: int = 3, seed: int | None = None):
        self.rows = rows
        self.cols = cols
        self.start = (0, 0)
        self.goal = (rows - 1, cols - 1)
        self._state = self.start
        # Seed is accepted for API consistency; this env is deterministic.

    # --- Gym-ish runtime API ---

    def reset(self, seed: int | None = None) -> tuple[int, int]:
        """Return the start state. Seed is accepted for API uniformity."""
        self._state = self.start
        return self._state

    def step(self, action: int) -> tuple[tuple[int, int], float, bool, dict]:
        """Advance one step. Returns (next_state, reward, done, info)."""
        if self.is_terminal(self._state):
            return self._state, 0.0, True, {}
        dr, dc = self._ACTION_DELTAS[action]
        r, c = self._state
        nr = max(0, min(self.rows - 1, r + dr))
        nc = max(0, min(self.cols - 1, c + dc))
        self._state = (nr, nc)
        reward = 0.0 if self.is_terminal(self._state) else -1.0
        return self._state, reward, self.is_terminal(self._state), {}

    def is_terminal(self, s: tuple[int, int]) -> bool:
        return s == self.goal

    # --- DP-friendly model API (unused in Phase 0, included for parity) ---

    def all_states(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def transitions(self, state, action):
        """Return list of (prob, next_state, reward) tuples.

        Deterministic env, so always a single outcome with prob 1.
        """
        if self.is_terminal(state):
            return [(1.0, state, 0.0)]
        dr, dc = self._ACTION_DELTAS[action]
        r, c = state
        nr = max(0, min(self.rows - 1, r + dr))
        nc = max(0, min(self.cols - 1, c + dc))
        ns = (nr, nc)
        reward = 0.0 if self.is_terminal(ns) else -1.0
        return [(1.0, ns, reward)]
