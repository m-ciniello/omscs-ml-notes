"""Q-networks: plain MLP and Dueling (Wang et al. 2016).
Dueling recombines as Q = V + (A - mean_a A) for identifiability.
Kept tiny — CartPole's 4-D state doesn't need depth."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.adv_head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)                 # (B, 1)
        a = self.adv_head(h)                   # (B, n_actions)
        return v + (a - a.mean(dim=-1, keepdim=True))


def build_q_network(
    *,
    state_dim: int,
    n_actions: int,
    hidden: int = 128,
    dueling: bool = False,
) -> nn.Module:
    cls = DuelingQNet if dueling else QNet
    return cls(state_dim=state_dim, n_actions=n_actions, hidden=hidden)
