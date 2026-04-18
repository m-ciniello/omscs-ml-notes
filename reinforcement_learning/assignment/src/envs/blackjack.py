"""Blackjack env + analytical MDP model.

This is the "DIY MDP" side of the assignment — the non-grid problem whose
transition function is defined explicitly so we can run VI and PI on it
without any sampling or model estimation.

The env matches Gymnasium's `Blackjack-v1` dynamics exactly (default rules:
natural=False, sab=False, infinite deck, dealer stands on all 17s). Older
Gym versions auto-hit the player to sum >= 12 on deal; modern Gymnasium
(>= 1.0) does not, so initial sums can be as low as 4. The state space is
therefore 18 * 10 * 2 = 360 non-terminal states plus a terminal sentinel.
(States with sum < 12 and usable_ace = 1 are combinatorially impossible,
but we include them for uniformity — they're simply never visited.)

Class surface:
    N_ACTIONS = 2                                  # 0 = stick, 1 = hit
    all_states()             -> list[State]        # MDP side
    is_terminal(state)       -> bool               #
    transitions(state, act)  -> list[(p, s', r)]   #
    reset(seed=None)         -> State              # rollout side (wraps Gym)
    step(action)             -> (s', r, done, info)

Two kinds of state:
- tuple (player_sum, dealer_card, usable_ace): non-terminal decision state
- ('terminal',) sentinel: absorbing state after the episode resolves

Why validate against Gymnasium rollouts at all? Because analytical MDPs are
notoriously fiddly (dealer soft-17 rules, ace arithmetic, natural bonuses).
A 5-line win-rate comparison against many Gym rollouts catches all of those
bugs in one shot. See `tests/validate_blackjack.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Card probabilities for an infinite deck.
# Ace = 1 (value 1 or 11 depending on context).
# 10 = 4/13 because J, Q, K all count as 10.
_CARD_PROBS: dict[int, float] = {
    1: 1 / 13,
    2: 1 / 13, 3: 1 / 13, 4: 1 / 13, 5: 1 / 13,
    6: 1 / 13, 7: 1 / 13, 8: 1 / 13, 9: 1 / 13,
    10: 4 / 13,
}

State = tuple  # either ('terminal',) or (player_sum, dealer_card, usable_ace)
TERMINAL: State = ("terminal",)


# ---------------------------------------------------------------------------
# Ace arithmetic
# ---------------------------------------------------------------------------

def _apply_card(psum: int, usable_ace: bool, card: int) -> tuple[int, bool]:
    """Add `card` to a hand summary (sum, has_usable_ace_flag).

    Encodes Blackjack's "greedy ace" rule: an ace counts as 11 whenever that
    doesn't bust; any existing usable ace is downgraded to 1 (i.e. sum -= 10)
    if a subsequent draw would otherwise bust. No-op if busting even after
    the downgrade (caller treats as a bust).
    """
    if card == 1:
        card_val = 11 if psum + 11 <= 21 else 1
        new_sum = psum + card_val
        new_ace = usable_ace or (card_val == 11)
    else:
        new_sum = psum + card
        new_ace = usable_ace

    if new_sum > 21 and new_ace:
        new_sum -= 10
        new_ace = False
    return new_sum, new_ace


# ---------------------------------------------------------------------------
# Dealer outcome distributions
# ---------------------------------------------------------------------------

def _dealer_playout_dist(start_sum: int, start_ace: bool) -> dict[int, float]:
    """Distribution over the dealer's final sum given (sum, ace) at decision time.

    Dealer policy: hit while sum_hand < 17 (stand on all 17s, including soft
    17 — this matches Gymnasium's Blackjack-v1 default). Bust is encoded as
    the sentinel sum = 22 (any value > 21 suffices; we pick 22 for readability).

    Memoised over (sum, ace) because the same intermediate hands appear
    across many calls.
    """
    memo: dict[tuple[int, bool], dict[int, float]] = {}

    def play(psum: int, ace: bool) -> dict[int, float]:
        key = (psum, ace)
        if key in memo:
            return memo[key]
        if psum >= 17:
            memo[key] = {psum: 1.0}
            return memo[key]

        dist: dict[int, float] = {}
        for card, prob in _CARD_PROBS.items():
            new_sum, new_ace = _apply_card(psum, ace, card)
            if new_sum > 21:
                sub = {22: 1.0}
            else:
                sub = play(new_sum, new_ace)
            for k, v in sub.items():
                dist[k] = dist.get(k, 0.0) + prob * v

        memo[key] = dist
        return dist

    return play(start_sum, start_ace)


def _dealer_dist_from_first_card(first_card: int) -> dict[int, float]:
    """Distribution over dealer final sum given only the visible first card.

    Marginalises over the dealer's hidden second card before the playout loop.
    """
    first_sum, first_ace = (11, True) if first_card == 1 else (first_card, False)

    dist: dict[int, float] = {}
    for second_card, prob in _CARD_PROBS.items():
        new_sum, new_ace = _apply_card(first_sum, first_ace, second_card)
        if new_sum > 21:
            sub = {22: 1.0}
        else:
            sub = _dealer_playout_dist(new_sum, new_ace)
        for k, v in sub.items():
            dist[k] = dist.get(k, 0.0) + prob * v
    return dist


# ---------------------------------------------------------------------------
# Main env class
# ---------------------------------------------------------------------------

class Blackjack:
    """Blackjack env with both analytical MDP and Gymnasium-backed rollouts."""

    N_ACTIONS = 2  # 0 = stick, 1 = hit
    ACTION_NAMES = ("stick", "hit")
    TERMINAL = TERMINAL

    def __init__(self, seed: int | None = None):
        # Precompute dealer playout distributions once per env instance.
        # Keyed by visible dealer card (1-10).
        self._dealer_dists: dict[int, dict[int, float]] = {
            c: _dealer_dist_from_first_card(c) for c in range(1, 11)
        }

        # Rollout side — lazy import so the package works without Gymnasium.
        self._gym_env = None
        self._rollout_seed = seed
        self._state: State = TERMINAL

    # ------------------------------------------------------------------
    # MDP model side (used by VI / PI)
    # ------------------------------------------------------------------

    def all_states(self) -> list[State]:
        """All non-terminal decision states + the terminal sentinel.

        Modern Gymnasium no longer auto-hits to 12, so we enumerate sums
        {4, ..., 21}. Some combinations (e.g. sum=4 with usable_ace=1) are
        combinatorially impossible but included for uniformity.
        """
        states: list[State] = []
        for psum in range(4, 22):
            for dcard in range(1, 11):
                for ace in (0, 1):
                    states.append((psum, dcard, ace))
        states.append(TERMINAL)
        return states

    def is_terminal(self, state: State) -> bool:
        return state == TERMINAL

    def transitions(self, state: State, action: int) -> list[tuple[float, State, float]]:
        """Exact (prob, next_state, reward) tuples for this (state, action).

        Because we use the TERMINAL sentinel, terminal-from-hit-bust and
        terminal-after-stick collapse into the same absorbing state; reward
        is carried on the transition, not on the next state.
        """
        if state == TERMINAL:
            return [(1.0, TERMINAL, 0.0)]

        psum, dcard, ace = state

        if action == 0:
            # Stick: dealer plays out; reward is the comparison of final sums.
            dealer_dist = self._dealer_dists[dcard]
            transitions = []
            for dealer_final, prob in dealer_dist.items():
                reward = _stick_reward(psum, dealer_final)
                transitions.append((prob, TERMINAL, reward))
            return transitions

        if action == 1:
            # Hit: draw a single card, resolve ace, possibly bust.
            transitions = []
            for card, prob in _CARD_PROBS.items():
                new_sum, new_ace = _apply_card(psum, bool(ace), card)
                if new_sum > 21:
                    transitions.append((prob, TERMINAL, -1.0))
                else:
                    transitions.append(
                        (prob, (new_sum, dcard, int(new_ace)), 0.0)
                    )
            return transitions

        raise ValueError(f"Invalid Blackjack action: {action}")

    # ------------------------------------------------------------------
    # Rollout side (used for evaluation and by model-free agents)
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> State:
        """Reset via Gymnasium's Blackjack-v1 for exact dynamics / init dist.

        Gymnasium is imported lazily so that DP-only workflows don't require
        the package.
        """
        if self._gym_env is None:
            import gymnasium as gym
            self._gym_env = gym.make("Blackjack-v1")

        reset_seed = seed if seed is not None else self._rollout_seed
        obs, _ = self._gym_env.reset(seed=reset_seed)
        self._rollout_seed = None  # consume the seed; subsequent resets are unseeded
        self._state = _obs_to_state(obs)
        return self._state

    def step(self, action: int) -> tuple[State, float, bool, dict]:
        if self._gym_env is None:
            raise RuntimeError(
                "Blackjack.step called before reset(). Call reset() first."
            )
        obs, reward, terminated, truncated, info = self._gym_env.step(int(action))
        done = bool(terminated or truncated)
        self._state = TERMINAL if done else _obs_to_state(obs)
        return self._state, float(reward), done, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stick_reward(player_sum: int, dealer_final: int) -> float:
    """+1 if player wins (including dealer bust), -1 if loses, 0 if ties."""
    if dealer_final > 21:  # dealer bust, player wins
        return 1.0
    if player_sum > dealer_final:
        return 1.0
    if player_sum < dealer_final:
        return -1.0
    return 0.0


def _obs_to_state(obs) -> State:
    """Convert a Gymnasium Blackjack observation tuple into our State type."""
    psum, dcard, ace = obs
    return (int(psum), int(dcard), int(ace))
