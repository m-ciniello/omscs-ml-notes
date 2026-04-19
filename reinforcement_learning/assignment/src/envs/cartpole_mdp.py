"""CartPole with an empirically-estimated MDP.

CartPole's ODE has no tabular transition model, so we build one from
rollouts. The FAQ sanctions this path. Evaluation uses the real dynamics,
not the estimate.

Estimator (maximum-likelihood counts, normalised once at the end):

    T̂(s' | s, a) = count(s, a -> s') / count(s, a)
    R̂(s, a)      = sum(rewards at (s, a)) / count(s, a)
    P̂(done | s, a) = count(terminating transitions at (s, a)) / count(s, a)

A transition `(s, a)` is emitted as `(p_term, TERMINAL, R̂)` plus one
`(count/total, s', R̂)` entry per observed next-state. Terminal absorption
is modelled explicitly (separate from the next-state histogram) so VI/PI
can zero out V(TERMINAL) correctly.

Notes:
- Sampling policy is pluggable (`build_action_fn`): uniform-random by
  default, or epsilon-greedy against a prior experiment's Q-table.
- Unvisited (s, a) -> TERMINAL with reward 0 (acts as a no-data prior).
  Leaving them as self-loops tends to produce pathological DP values.
- Each seed builds its own MDP from a fresh sampling stream, so multi-seed
  runs capture model-estimation variance in the DP-derived policy.
"""

from __future__ import annotations

import pickle
from typing import Any, Callable

import numpy as np

from src.envs.cartpole import (
    _DEFAULT_BOUNDS,
    _DEFAULT_N_BINS,
    DiscretizedCartPole,
)


TERMINAL: tuple = ("terminal",)   # sentinel state for absorbing DP

# Type alias: a sampling policy is a function (state, rng) -> action_index.
ActionFn = Callable[[tuple, np.random.Generator], int]


class MDPEstimate:
    """Counts + reward sums from rollout-based model estimation.
    Normalised once at the end by the env constructor."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions
        self.visit_counts: dict[tuple, np.ndarray] = {}        # s -> (n_actions,)
        self.reward_sums: dict[tuple, np.ndarray] = {}         # s -> (n_actions,)
        self.next_state_counts: dict[tuple, list[dict]] = {}   # s -> [dict(s'->count), ...]
        self.termination_counts: dict[tuple, np.ndarray] = {}  # s -> (n_actions,)

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        if state not in self.visit_counts:
            self.visit_counts[state] = np.zeros(self.n_actions, dtype=np.int64)
            self.reward_sums[state] = np.zeros(self.n_actions, dtype=np.float64)
            self.next_state_counts[state] = [
                {} for _ in range(self.n_actions)
            ]
            self.termination_counts[state] = np.zeros(self.n_actions, dtype=np.int64)

        self.visit_counts[state][action] += 1
        self.reward_sums[state][action] += reward

        if done:
            self.termination_counts[state][action] += 1
            # post-done next_state is undefined in Gym; skip recording it
        else:
            bucket = self.next_state_counts[state][action]
            bucket[next_state] = bucket.get(next_state, 0) + 1


def estimate_mdp(
    base_env: DiscretizedCartPole,
    *,
    n_sampling_episodes: int,
    sampling_seed: int,
    action_fn: ActionFn,
    max_steps_per_episode: int = 500,
) -> MDPEstimate:
    """Roll out episodes and accumulate (counts, reward sums, termination counts).
    The action_fn is injected so the estimator doesn't care whether sampling
    is random, ε-greedy on a trained Q, etc."""
    rng = np.random.default_rng(sampling_seed)
    est = MDPEstimate(n_actions=base_env.N_ACTIONS)

    for ep in range(n_sampling_episodes):
        # deterministic per-episode seed so re-runs give byte-identical counts
        episode_seed = int(rng.integers(0, 2**31 - 1))
        state = base_env.reset(seed=episode_seed)
        for _ in range(max_steps_per_episode):
            action = action_fn(state, rng)
            next_state, reward, done, _ = base_env.step(action)
            est.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state

    return est


def build_action_fn(
    *,
    policy: str,
    n_actions: int,
    source_experiment: str | None = None,
    epsilon: float = 0.2,
    source_seed_index: int = 0,
) -> ActionFn:
    """Build a (state, rng) -> action callable. Supported: "random" or
    "epsilon_greedy" (loads a Q-table from a prior experiment's seed dir)."""
    if policy == "random":
        def random_fn(state: tuple, rng: np.random.Generator) -> int:
            return int(rng.integers(0, n_actions))
        return random_fn

    if policy == "epsilon_greedy":
        if source_experiment is None:
            raise ValueError(
                "policy='epsilon_greedy' requires source_experiment."
            )
        from src.experiments.runner import experiment_dir  # lazy
        exp_dir = experiment_dir(source_experiment)
        seed_dirs = sorted(exp_dir.glob("seed_*"))
        if not seed_dirs:
            raise FileNotFoundError(
                f"No seed_* directories under {exp_dir}. Train the sampling-"
                f"policy source experiment ({source_experiment!r}) first."
            )
        picked = seed_dirs[source_seed_index % len(seed_dirs)]
        with open(picked / "result.pkl", "rb") as f:
            Q = pickle.load(f).get("Q")
        if Q is None:
            raise ValueError(
                f"Experiment {source_experiment!r} seed {picked.name} has no "
                f"Q-table in its result.pkl. Not a tabular agent?"
            )
        eps = float(epsilon)

        def eps_greedy_fn(state: tuple, rng: np.random.Generator) -> int:
            if rng.random() < eps or state not in Q:
                return int(rng.integers(0, n_actions))
            return int(np.argmax(Q[state]))
        return eps_greedy_fn

    raise ValueError(f"Unknown sampling policy {policy!r}")


class CartPoleEstimatedMDP:
    """CartPole with an empirically-estimated transition model for VI/PI,
    and real-dynamics rollouts for evaluation."""

    N_ACTIONS = 2
    ACTION_NAMES = ("left", "right")

    def __init__(
        self,
        *,
        n_bins: tuple[int, ...] = _DEFAULT_N_BINS,
        bounds: tuple[tuple[float, float], ...] = _DEFAULT_BOUNDS,
        n_sampling_episodes: int = 5_000,
        sampling_seed: int | None = None,
        sampling_policy: str = "random",
        sampling_source_experiment: str | None = None,
        sampling_epsilon: float = 0.2,
        max_steps_per_episode: int = 500,
        seed: int | None = None,
    ):
        # fall back to the runner's per-seed seed so multi-seed runs produce
        # distinct estimated MDPs (variance in the DP-derived policy)
        if sampling_seed is None:
            sampling_seed = 0 if seed is None else int(seed)

        self.n_bins = tuple(int(n) for n in n_bins)
        self.bounds = tuple((float(lo), float(hi)) for lo, hi in bounds)
        self.n_sampling_episodes = int(n_sampling_episodes)
        self.sampling_seed = int(sampling_seed)
        self.sampling_policy = sampling_policy
        self.sampling_source_experiment = sampling_source_experiment
        self.sampling_epsilon = float(sampling_epsilon)
        self.max_steps_per_episode = int(max_steps_per_episode)

        # Eval-side env: real dynamics, used for reset/step during rollouts.
        self._rollout_env = DiscretizedCartPole(
            n_bins=self.n_bins, bounds=self.bounds, seed=seed,
        )

        # Sampling-side env: a separate instance so the eval seed stream
        # isn't tangled up with sampling. Same class, same discretization.
        self._sampling_env = DiscretizedCartPole(
            n_bins=self.n_bins, bounds=self.bounds, seed=sampling_seed,
        )

        action_fn = build_action_fn(
            policy=self.sampling_policy,
            n_actions=self.N_ACTIONS,
            source_experiment=self.sampling_source_experiment,
            epsilon=self.sampling_epsilon,
            # Use this env's sampling_seed to pick which source-experiment
            # seed's Q-table to load. With matched seed counts across both
            # experiments this preserves the "N independent samples" story.
            source_seed_index=self.sampling_seed,
        )

        est = estimate_mdp(
            self._sampling_env,
            n_sampling_episodes=self.n_sampling_episodes,
            sampling_seed=self.sampling_seed,
            action_fn=action_fn,
            max_steps_per_episode=self.max_steps_per_episode,
        )
        self._estimate = est

        # Precompute `transitions(s, a)` lookup tables once.
        self._transitions: dict[tuple[tuple, int], list[tuple[float, tuple, float]]] = {}
        self._states: list[tuple] = [TERMINAL]
        self._states_set: set[tuple] = {TERMINAL}
        for s, counts in est.visit_counts.items():
            if s not in self._states_set:
                self._states.append(s)
                self._states_set.add(s)
            for a in range(self.N_ACTIONS):
                total = int(counts[a])
                if total == 0:
                    self._transitions[(s, a)] = [
                        (1.0, TERMINAL, 0.0)  # unseen => go to terminal with 0
                    ]
                    continue
                mean_r = float(est.reward_sums[s][a] / total)
                n_term = int(est.termination_counts[s][a])
                p_term = n_term / total
                triples: list[tuple[float, tuple, float]] = []
                if p_term > 0:
                    triples.append((p_term, TERMINAL, mean_r))
                for ns, cnt in est.next_state_counts[s][a].items():
                    triples.append((cnt / total, ns, mean_r))
                self._transitions[(s, a)] = triples

    # ---------- MDP model side ----------------------------------------

    def all_states(self):
        return list(self._states)

    def is_terminal(self, state) -> bool:
        return state == TERMINAL

    def transitions(self, state, action: int) -> list[tuple[float, tuple, float]]:
        if state == TERMINAL:
            return [(1.0, TERMINAL, 0.0)]
        return self._transitions.get((state, int(action)),
                                     [(1.0, TERMINAL, 0.0)])

    # ---------- Rollout side ------------------------------------------

    def reset(self, seed: int | None = None):
        return self._rollout_env.reset(seed=seed)

    def step(self, action: int):
        return self._rollout_env.step(action)

    # ---------- Diagnostics -------------------------------------------

    def coverage_stats(self) -> dict[str, Any]:
        """How well did sampling cover the discrete state space?"""
        n_total_states = 1
        for n in self.n_bins:
            n_total_states *= n
        n_visited_states = len(self._states) - 1  # exclude TERMINAL
        total_transitions = int(sum(
            int(c.sum()) for c in self._estimate.visit_counts.values()
        ))
        total_sa_pairs_visited = int(sum(
            int((c > 0).sum()) for c in self._estimate.visit_counts.values()
        ))
        total_sa_pairs_possible = n_visited_states * self.N_ACTIONS
        return {
            "n_sampling_episodes": self.n_sampling_episodes,
            "total_transitions_sampled": total_transitions,
            "n_states_possible": n_total_states,
            "n_states_visited": n_visited_states,
            "state_coverage_fraction": n_visited_states / max(n_total_states, 1),
            "n_sa_pairs_visited": total_sa_pairs_visited,
            "n_sa_pairs_possible_given_visited_states": total_sa_pairs_possible,
        }
