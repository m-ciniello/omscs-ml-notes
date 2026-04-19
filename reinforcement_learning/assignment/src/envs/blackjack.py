"""Blackjack-v1 with both an analytical MDP and Gymnasium-backed rollouts.

Matches Gymnasium's default Blackjack-v1 dynamics (natural=False, sab=False,
infinite deck, dealer stands on 17). Non-terminal states are
(player_sum, dealer_card, usable_ace); the TERMINAL sentinel absorbs
everything after a hand resolves.
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


def _apply_card(psum: int, usable_ace: bool, card: int) -> tuple[int, bool]:
    """Add `card` to (sum, usable_ace_flag), applying the greedy-ace rule."""
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


def _dealer_playout_dist(start_sum: int, start_ace: bool) -> dict[int, float]:
    """Distribution over the dealer's final sum. Dealer hits while < 17
    (stand on 17, including soft 17). Bust encoded as sum = 22. Memoised."""
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
    """Dealer-final-sum distribution from the visible first card, marginalising
    over the hidden second card."""
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


class Blackjack:
    """Blackjack env with both analytical MDP and Gymnasium-backed rollouts."""

    N_ACTIONS = 2  # 0 = stick, 1 = hit
    ACTION_NAMES = ("stick", "hit")
    TERMINAL = TERMINAL

    def __init__(self, seed: int | None = None):
        # Precompute dealer playout dists once per env, keyed by visible card.
        self._dealer_dists: dict[int, dict[int, float]] = {
            c: _dealer_dist_from_first_card(c) for c in range(1, 11)
        }
        self._gym_env = None  # lazy import so the module works without Gym
        self._rollout_seed = seed
        self._state: State = TERMINAL

    # --- MDP model side (VI / PI) ---

    def all_states(self) -> list[State]:
        """All decision states (sums 4..21 x dealer 1..10 x ace 0,1) + TERMINAL.
        Some combos (e.g. sum=4 usable_ace=1) are impossible but harmless."""
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
        """Exact (prob, next_state, reward) for (state, action). Reward rides
        the transition, not the next state, so bust-on-hit and stick-then-resolve
        both absorb to the same TERMINAL sentinel."""
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

    # --- Rollout side (evaluation + model-free agents) ---

    def reset(self, seed: int | None = None) -> State:
        """Reset via Gymnasium's Blackjack-v1 for exact dynamics."""
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


def _stick_reward(player_sum: int, dealer_final: int) -> float:
    """+1 if player wins (incl. dealer bust), -1 if loses, 0 on tie."""
    if dealer_final > 21:  # dealer bust, player wins
        return 1.0
    if player_sum > dealer_final:
        return 1.0
    if player_sum < dealer_final:
        return -1.0
    return 0.0


def _obs_to_state(obs) -> State:
    psum, dcard, ace = obs
    return (int(psum), int(dcard), int(ace))
