"""Q-Learning — off-policy TD(0): Q(s,a) <- Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)].
Target uses the greedy action at s' regardless of the ε-greedy behaviour
policy, so Q converges toward Q* while exploring. Mirrors SARSA line-for-line
except for the TD target."""

from __future__ import annotations

import time

import numpy as np

from src.agents.tabular import (
    QTable,
    epsilon_greedy,
    eval_greedy_policy,
    linear_epsilon,
)


class QLearning:
    name = "qlearning"

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 10_000,
        max_steps_per_episode: int = 500,
    ):
        self.alpha = float(alpha)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_episodes = int(epsilon_decay_episodes)
        self.max_steps_per_episode = int(max_steps_per_episode)

    def run(
        self,
        env,
        *,
        n_episodes: int,
        eval_episodes: int,
        gamma: float,
        seed: int,
    ) -> dict:
        rng = np.random.default_rng(seed)
        qtable = QTable(n_actions=env.N_ACTIONS)

        train_returns: list[float] = []
        train_steps: list[int] = []
        epsilon_history: list[float] = []

        t0 = time.perf_counter()

        for ep in range(n_episodes):
            eps = linear_epsilon(
                ep,
                start=self.epsilon_start,
                end=self.epsilon_end,
                decay_episodes=self.epsilon_decay_episodes,
            )
            epsilon_history.append(eps)

            state = env.reset()
            total_return = 0.0
            ep_steps = self.max_steps_per_episode
            for t in range(self.max_steps_per_episode):
                action = epsilon_greedy(qtable, state, eps, rng)
                next_state, reward, done, _ = env.step(action)
                total_return += reward

                q_row = qtable.values(state)
                if done:
                    td_target = reward
                else:
                    td_target = reward + gamma * qtable.values(next_state).max()

                q_row[action] += self.alpha * (td_target - q_row[action])

                if done:
                    ep_steps = t + 1
                    break

                state = next_state

            train_steps.append(ep_steps)
            train_returns.append(total_return)

        train_wall = time.perf_counter() - t0

        eval_returns, eval_steps = eval_greedy_policy(
            env,
            qtable,
            eval_episodes=eval_episodes,
            max_steps_per_episode=self.max_steps_per_episode,
        )

        return {
            "train_returns": train_returns,
            "train_steps": train_steps,
            "eval_returns": eval_returns,
            "eval_steps": eval_steps,
            "history": {
                "epsilon_per_episode": epsilon_history,
                "n_visited_states": qtable.n_visited_states(),
                "train_wall_seconds": train_wall,
            },
            "policy": qtable.policy_dict(),
            "Q": qtable.to_dict(),
            "wall_clock_seconds": time.perf_counter() - t0,
        }
