"""Gymnasium CartPole-v1 wrapped with binned state discretization.

For DP on CartPole, see `cartpole_mdp.py` (builds a tabular MDP from
rollouts). Default binning (3,3,8,12) = 864 non-terminal states — angle and
angular velocity get finer bins because they dominate the control problem.
This matches the FAQ starter grid and is the grid the `src/configs/`
package uses for all single-point CartPole experiments. Bounds match the FAQ clamps
([-2.4, 2.4], [-3, 3], [-0.2, 0.2], [-3.5, 3.5]); rare out-of-bounds
values are clipped to the nearest edge (those transitions typically
terminate the episode anyway).
"""

from __future__ import annotations

from typing import Any

import numpy as np


State = tuple  # tuple of bin indices, one per observation dim

_DEFAULT_BOUNDS = (
    (-2.4, 2.4),    # cart_position
    (-3.0, 3.0),    # cart_velocity
    (-0.2, 0.2),    # pole_angle (radians)
    (-3.5, 3.5),    # pole_angular_velocity
)
_DEFAULT_N_BINS = (3, 3, 8, 12)


def _discretize(
    obs,
    bounds: tuple[tuple[float, float], ...],
    n_bins: tuple[int, ...],
) -> State:
    """Map a 4-D continuous obs to a tuple of bin indices. Out-of-range
    values clip to the nearest edge; n_bins == 1 collapses that dim."""
    state = []
    for x, (lo, hi), n in zip(obs, bounds, n_bins):
        if n <= 1:
            state.append(0)
            continue
        x_clipped = lo if x < lo else hi if x > hi else x
        frac = (x_clipped - lo) / (hi - lo)
        idx = int(frac * n)
        if idx >= n:
            idx = n - 1
        state.append(idx)
    return tuple(state)


class DiscretizedCartPole:
    """CartPole-v1 with binned state. Interface: N_ACTIONS, reset(seed)->State,
    step(a)->(State, reward, done, info). Gym is imported lazily."""

    N_ACTIONS = 2
    ACTION_NAMES = ("left", "right")

    def __init__(
        self,
        *,
        n_bins: tuple[int, ...] = _DEFAULT_N_BINS,
        bounds: tuple[tuple[float, float], ...] = _DEFAULT_BOUNDS,
        seed: int | None = None,
    ):
        if len(n_bins) != 4 or len(bounds) != 4:
            raise ValueError(
                f"CartPole has 4 obs dims; got n_bins={n_bins}, bounds={bounds}."
            )
        self.n_bins = tuple(int(n) for n in n_bins)
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)

        self._gym_env = None
        self._rollout_seed = seed
        self._state: State | None = None

    def reset(self, seed: int | None = None) -> State:
        if self._gym_env is None:
            import gymnasium as gym
            self._gym_env = gym.make("CartPole-v1")

        reset_seed = seed if seed is not None else self._rollout_seed
        obs, _ = self._gym_env.reset(seed=reset_seed)
        self._rollout_seed = None  # consume seed; subsequent resets are unseeded
        self._state = _discretize(obs, self.bounds, self.n_bins)
        return self._state

    def step(self, action: int) -> tuple[State, float, bool, dict[str, Any]]:
        if self._gym_env is None:
            raise RuntimeError(
                "DiscretizedCartPole.step called before reset(). Call reset() first."
            )
        obs, reward, terminated, truncated, info = self._gym_env.step(int(action))
        done = bool(terminated or truncated)
        self._state = _discretize(obs, self.bounds, self.n_bins)
        return self._state, float(reward), done, info

    def n_total_states(self) -> int:
        total = 1
        for n in self.n_bins:
            total *= n
        return total


class ContinuousCartPole:
    """CartPole-v1 passthrough: raw 4-D float obs for DQN/Rainbow."""

    N_ACTIONS = 2
    ACTION_NAMES = ("left", "right")
    STATE_DIM = 4

    def __init__(self, *, seed: int | None = None):
        self._gym_env = None
        self._rollout_seed = seed

    def reset(self, seed: int | None = None):
        if self._gym_env is None:
            import gymnasium as gym
            self._gym_env = gym.make("CartPole-v1")
        reset_seed = seed if seed is not None else self._rollout_seed
        obs, _ = self._gym_env.reset(seed=reset_seed)
        self._rollout_seed = None
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: int):
        if self._gym_env is None:
            raise RuntimeError(
                "ContinuousCartPole.step called before reset(). Call reset() first."
            )
        obs, reward, terminated, truncated, info = self._gym_env.step(int(action))
        done = bool(terminated or truncated)
        return np.asarray(obs, dtype=np.float32), float(reward), done, info
