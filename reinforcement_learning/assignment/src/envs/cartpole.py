"""CartPole-v1 wrapped with binned state discretization.

The assignment asks for a second environment with different structure from
Blackjack. CartPole gives us exactly that: continuous 4-D observations, a
1-step reward of +1 per survived step, and truncation at 500 steps. To keep
the same tabular agents working, we discretize the observation into integer
bin tuples — the same state representation the rest of the codebase expects.

Why binned (not tile coding)?
    - Simplest discretization that produces a genuine Markov-ish state for
      coarse bins. Enough to demonstrate SARSA / Q-Learning learning curves,
      sample complexity, and gamma effects — the phenomena the report
      actually needs to discuss.
    - Tile coding would be strictly more powerful but adds function-approx
      machinery we don't want to explain alongside the tabular story. We'll
      cross the function-approx bridge with DQN in Phase 4.

No MDP model side (`all_states` / `transitions`) is exposed. CartPole's
physics integration is nontrivial, and an *estimated* MDP model from rollouts
would be a different project. The DP comparison on this env is therefore
intentionally out of scope — this is the "rollout-only" env that contrasts
with Blackjack's analytical MDP.

Default discretization: n_bins = (3, 3, 6, 6) = 324 non-terminal states.
    - cart_position:         3 bins over [-2.4, 2.4]
    - cart_velocity:         3 bins over [-3.0, 3.0]
    - pole_angle:            6 bins over [-0.21, 0.21]   (±12°; termination
                                                          bound is ±0.2095)
    - pole_angular_velocity: 6 bins over [-3.5, 3.5]

Angle + angular velocity dominate the control problem (cart position and
velocity are secondary), which is why those two get finer bins by default.
Bounds are picked to exceed the practical range of stable trajectories but
not so wide that most bins are unused; slight over-clipping of extreme
values is fine because those transitions are rare and end the episode.
"""

from __future__ import annotations

from typing import Any


State = tuple  # tuple of bin indices, one per observation dim

_DEFAULT_BOUNDS = (
    (-2.4, 2.4),    # cart_position
    (-3.0, 3.0),    # cart_velocity
    (-0.21, 0.21),  # pole_angle (radians)
    (-3.5, 3.5),    # pole_angular_velocity
)
_DEFAULT_N_BINS = (3, 3, 6, 6)


def _discretize(
    obs,
    bounds: tuple[tuple[float, float], ...],
    n_bins: tuple[int, ...],
) -> State:
    """Map a 4-D continuous obs to a tuple of bin indices.

    Values outside `bounds` are clipped to the nearest edge. Bin 0 covers
    [lo, lo + w), bin 1 covers [lo + w, lo + 2w), ..., bin n-1 covers
    [lo + (n-1)w, hi] (i.e. the upper edge lands in the last bin). A
    dimension with n_bins == 1 collapses to a single bucket regardless of
    value.
    """
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
    """Gymnasium CartPole-v1 with binned state discretization.

    Exposes the same minimal interface used by RandomAgent, SARSA, Q-Learning:
        N_ACTIONS:              class attribute, 2
        reset(seed=None):       -> State
        step(action):           -> (State, reward, done, info)

    The Gym env is imported lazily so the module can be imported without
    Gymnasium installed (mirrors Blackjack's pattern).
    """

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

    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Introspection (handy for reporting / plots)
    # ------------------------------------------------------------------

    def n_total_states(self) -> int:
        total = 1
        for n in self.n_bins:
            total *= n
        return total
