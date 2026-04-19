"""Replay buffers: uniform (ring-buffer) and prioritized (sum-tree, Schaul 2016).

Common interface: push(s,a,r,s',done); sample(B, rng) -> Batch;
update_priorities(indices, td_errors) (PER only; no-op for uniform).
PER priorities are proportional: p_i = (|td| + eps)**alpha. IS weights
w_i = (N * P(i))**-beta / max_w; beta is annealed by the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class Batch(NamedTuple):
    """Sampled minibatch. For uniform sampling, `weights` is all-ones."""
    states: np.ndarray      # (B, state_dim) float32
    actions: np.ndarray     # (B,) int64
    rewards: np.ndarray     # (B,) float32
    next_states: np.ndarray # (B, state_dim) float32
    dones: np.ndarray       # (B,) float32 (0 / 1)
    indices: np.ndarray     # (B,) int64
    weights: np.ndarray     # (B,) float32


class UniformReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)
        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._size

    def push(self, s, a, r, s_next, done) -> None:
        i = self._pos
        self.states[i]      = s
        self.actions[i]     = a
        self.rewards[i]     = r
        self.next_states[i] = s_next
        self.dones[i]       = float(done)
        self._pos = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        idx = rng.integers(0, self._size, size=batch_size)
        return Batch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx],
            indices=idx.astype(np.int64),
            weights=np.ones(batch_size, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors) -> None:
        pass  # no-op; API compatibility with PER


class _SumTree:
    """Binary sum-tree: sample a leaf in O(log N) with probability proportional
    to its priority. Flat array of 2*capacity; internal nodes 0..capacity-1,
    leaves at capacity..2*capacity-1 (leaf i -> tree[i + capacity])."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1]) if self.capacity > 0 else 0.0

    def set_priority(self, leaf_idx: int, priority: float) -> None:
        node = leaf_idx + self.capacity
        delta = priority - self.tree[node]
        self.tree[node] = priority
        node //= 2
        while node >= 1:
            self.tree[node] += delta
            node //= 2

    def sample(self, value: float) -> int:
        """Return the leaf index whose cumulative sum covers ``value``."""
        node = 1
        while node < self.capacity:
            left = 2 * node
            if value <= self.tree[left]:
                node = left
            else:
                value -= self.tree[left]
                node = left + 1
        return node - self.capacity


class PrioritizedReplayBuffer:
    """Proportional PER (Schaul 2016). Capacity rounds up to a power of two
    so the sum-tree is complete; unused leaves have priority 0."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        *,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        self.user_capacity = int(capacity)   # "effective" capacity for push

        self.states      = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(self.capacity, dtype=np.int64)
        self.rewards     = np.zeros(self.capacity, dtype=np.float32)
        self.dones       = np.zeros(self.capacity, dtype=np.float32)

        self.tree = _SumTree(self.capacity)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._max_priority = 1.0

        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return self._size

    def push(self, s, a, r, s_next, done) -> None:
        i = self._pos
        self.states[i]      = s
        self.actions[i]     = a
        self.rewards[i]     = r
        self.next_states[i] = s_next
        self.dones[i]       = float(done)
        self.tree.set_priority(i, (self._max_priority + self.eps) ** self.alpha)
        self._pos = (i + 1) % self.user_capacity
        self._size = min(self._size + 1, self.user_capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, *, beta: float = 0.4) -> Batch:
        total = self.tree.total
        # Stratified sampling: batch_size equal-width segments of the sum.
        segment = total / batch_size
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        for b in range(batch_size):
            lo, hi = segment * b, segment * (b + 1)
            v = rng.uniform(lo, hi)
            leaf = self.tree.sample(v)
            # Clamp to valid stored range in case of floating-point edges.
            leaf = min(leaf, self._size - 1)
            indices[b] = leaf
            priorities[b] = self.tree.tree[leaf + self.capacity]

        probs = priorities / max(total, 1e-12)
        weights = (self._size * probs) ** (-beta)
        weights = weights / max(weights.max(), 1e-12)

        return Batch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
            indices=indices,
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors) -> None:
        abs_err = np.abs(np.asarray(td_errors, dtype=np.float64)) + self.eps
        new_p = abs_err ** self.alpha
        for idx, p in zip(indices, new_p):
            self.tree.set_priority(int(idx), float(p))
        self._max_priority = max(self._max_priority, float(abs_err.max()))
