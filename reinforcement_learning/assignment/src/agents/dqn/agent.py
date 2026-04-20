"""DQN with optional Double / Dueling / PER / N-step.

Four orthogonal flags (double, dueling, per, nstep) drive the Rainbow-medium
ablation registered in `src/configs/dqn_ablation.py`. All four on is
"Rainbow-medium" (minus C51 and NoisyNets, de-scoped up front).
"""

from __future__ import annotations

import copy
import time
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.dqn.network import build_q_network
from src.agents.dqn.replay import (
    PrioritizedReplayBuffer,
    UniformReplayBuffer,
)


class DQNAgent:
    """DQN with optional Double / Dueling / PER / N-step. All HPs default to
    sensible CartPole values; see `src/configs/dqn_ablation.py` for the
    ablation variants."""

    name = "dqn"

    def __init__(self, **hp: Any):
        self.hidden              = int(hp.get("hidden", 128))
        self.lr                  = float(hp.get("lr", 1e-3))
        self.buffer_capacity     = int(hp.get("buffer_capacity", 10_000))
        self.batch_size          = int(hp.get("batch_size", 64))
        self.warmup_steps        = int(hp.get("warmup_steps", 500))
        self.train_freq          = int(hp.get("train_freq", 1))
        self.target_update_freq  = int(hp.get("target_update_freq", 500))
        self.grad_clip: float | None = hp.get("grad_clip", 10.0)

        # Exploration
        self.eps_start           = float(hp.get("eps_start", 1.0))
        self.eps_end             = float(hp.get("eps_end", 0.05))
        self.eps_decay_steps     = int(hp.get("eps_decay_steps", 10_000))

        # Component toggles
        self.use_double          = bool(hp.get("double", False))
        self.use_dueling         = bool(hp.get("dueling", False))
        self.use_per             = bool(hp.get("per", False))
        self.per_alpha           = float(hp.get("per_alpha", 0.6))
        self.per_beta_start      = float(hp.get("per_beta_start", 0.4))
        self.per_beta_end        = float(hp.get("per_beta_end", 1.0))
        self.per_beta_steps      = int(hp.get("per_beta_steps", 20_000))

        self.nstep               = int(hp.get("nstep", 1))
        if self.nstep < 1:
            raise ValueError(f"nstep must be >= 1, got {self.nstep}")

        self.hyperparams = hp
        self.device = torch.device("cpu")  # CartPole-v1: GPU doesn't help

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
        torch.manual_seed(seed)

        state_dim = _infer_state_dim(env)
        n_actions = int(env.N_ACTIONS)

        # Networks
        q_net = build_q_network(
            state_dim=state_dim, n_actions=n_actions,
            hidden=self.hidden, dueling=self.use_dueling,
        ).to(self.device)
        target_net = copy.deepcopy(q_net)
        for p in target_net.parameters():
            p.requires_grad_(False)
        optim = torch.optim.Adam(q_net.parameters(), lr=self.lr)

        # Replay
        if self.use_per:
            buf = PrioritizedReplayBuffer(
                capacity=self.buffer_capacity, state_dim=state_dim,
                alpha=self.per_alpha,
            )
        else:
            buf = UniformReplayBuffer(
                capacity=self.buffer_capacity, state_dim=state_dim,
            )

        # N-step transition queue (only used if self.nstep > 1)
        nstep_buf: deque = deque(maxlen=self.nstep)

        train_returns: list[float] = []
        train_steps: list[int] = []
        losses: list[float] = []
        eps_trajectory: list[float] = []

        global_step = 0
        wall_start = time.perf_counter()
        for ep in range(n_episodes):
            state = env.reset(seed=seed + ep if ep == 0 else None)
            ep_return = 0.0
            ep_steps = 0
            nstep_buf.clear()

            done = False
            while not done:
                eps = _linear_schedule(
                    global_step, self.eps_start, self.eps_end, self.eps_decay_steps,
                )
                action = self._select_action(q_net, state, n_actions, eps, rng)
                next_state, reward, done, _ = env.step(action)
                ep_return += reward
                ep_steps += 1
                global_step += 1

                # Push (possibly n-step-aggregated) transition to replay.
                self._push_transition(
                    buf, nstep_buf, state, action, reward, next_state, done, gamma,
                )
                state = next_state

                # Train.
                if (
                    len(buf) >= self.warmup_steps
                    and len(buf) >= self.batch_size
                    and global_step % self.train_freq == 0
                ):
                    beta = _linear_schedule(
                        global_step, self.per_beta_start, self.per_beta_end,
                        self.per_beta_steps,
                    )
                    loss = self._train_step(
                        q_net, target_net, optim, buf, gamma, beta, rng,
                    )
                    losses.append(loss)

                # Hard target update.
                if global_step % self.target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

            eps_trajectory.append(eps)
            train_returns.append(float(ep_return))
            train_steps.append(int(ep_steps))

        wall = time.perf_counter() - wall_start

        # ---- Evaluation (greedy rollouts) ----
        eval_returns, eval_steps = [], []
        for i in range(eval_episodes):
            s = env.reset(seed=seed + 10_000 + i)
            r_total, n_t = 0.0, 0
            done = False
            while not done:
                with torch.no_grad():
                    q = q_net(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))
                a = int(q.argmax(dim=1).item())
                s, r, done, _ = env.step(a)
                r_total += r
                n_t += 1
            eval_returns.append(float(r_total))
            eval_steps.append(int(n_t))

        return {
            "train_returns": train_returns,
            "train_steps": train_steps,
            "eval_returns": eval_returns,
            "eval_steps": eval_steps,
            "history": {
                "losses": losses,
                "epsilon_per_episode": eps_trajectory,
                "global_steps": global_step,
            },
            "policy": None,                 # continuous state; no tabular policy
            "Q": None,                      # torch module, not serialised here
            "wall_clock_seconds": float(wall),
        }

    def _select_action(self, q_net, state, n_actions: int, eps: float, rng) -> int:
        if rng.random() < eps:
            return int(rng.integers(0, n_actions))
        with torch.no_grad():
            q = q_net(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def _push_transition(
        self, buf, nstep_buf, state, action, reward, next_state, done, gamma,
    ) -> None:
        """Push a 1-step or n-step aggregated transition to the replay buffer.

        For n=1 this is just a direct push. For n>1, we accumulate the last
        n single-step transitions and emit an n-step transition once the
        queue is full (or the episode ends, in which case we flush what
        we have at shorter n).
        """
        if self.nstep == 1:
            buf.push(state, action, reward, next_state, done)
            return

        nstep_buf.append((state, action, reward, next_state, done))

        # While the episode is ongoing, emit exactly one full n-step
        # transition whenever the deque fills up. The deque's ``maxlen=nstep``
        # auto-pops the oldest entry on the next append, so no explicit
        # popleft is needed here. On episode end, the drain loop below
        # emits starting from the head (same transition this branch would
        # have produced), then shorter-horizon tails — so triggering both
        # would duplicate the terminal-step n-step transition in replay
        # (one per terminal). Guarding on ``not done`` eliminates that.
        if not done and len(nstep_buf) == self.nstep:
            self._emit_nstep(buf, nstep_buf, gamma)

        # On episode end, drain the deque at progressively shorter horizons.
        if done:
            while nstep_buf:
                self._emit_nstep(buf, nstep_buf, gamma)
                nstep_buf.popleft()

    def _emit_nstep(self, buf, nstep_buf, gamma: float) -> None:
        """Emit one transition from the head of nstep_buf.

        Aggregates the k remaining transitions (k <= self.nstep) starting
        from the head, giving a k-step return and a bootstrap target off
        the last next_state (or a done-flag if the episode ended inside).
        """
        n = len(nstep_buf)
        s0, a0, _, _, _ = nstep_buf[0]
        cum_r = 0.0
        done_final = False
        next_final = None
        for i, (_, _, r, sn, d) in enumerate(nstep_buf):
            cum_r += (gamma ** i) * r
            next_final = sn
            if d:
                done_final = True
                break
        buf.push(s0, a0, cum_r, next_final, done_final)

    def _train_step(
        self, q_net, target_net, optim, buf, gamma: float, beta: float, rng,
    ) -> float:
        """One gradient step on a minibatch drawn from ``buf``.

        Returns the scalar MSE (PER: weighted MSE) for logging. The target
        uses the N-step-effective discount ``gamma ** n`` when n > 1 — this
        is important because transitions already carry an n-step cumulative
        reward.
        """
        if self.use_per:
            batch = buf.sample(self.batch_size, rng, beta=beta)
        else:
            batch = buf.sample(self.batch_size, rng)

        s      = torch.as_tensor(batch.states)
        a      = torch.as_tensor(batch.actions)
        r      = torch.as_tensor(batch.rewards)
        s_next = torch.as_tensor(batch.next_states)
        done   = torch.as_tensor(batch.dones)
        w      = torch.as_tensor(batch.weights)

        q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double:
                next_actions = q_net(s_next).argmax(dim=1, keepdim=True)
                next_q = target_net(s_next).gather(1, next_actions).squeeze(1)
            else:
                next_q = target_net(s_next).max(dim=1).values
            discount = gamma ** self.nstep
            target = r + discount * next_q * (1.0 - done)

        td_err = target - q_sa
        # PER: importance-sampling-weighted Huber loss is more stable than MSE,
        # but for CartPole-v1 plain weighted MSE works fine.
        loss = (w * td_err.pow(2)).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), self.grad_clip)
        optim.step()

        if self.use_per:
            buf.update_priorities(batch.indices, td_err.detach().cpu().numpy())

        return float(loss.item())


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _infer_state_dim(env) -> int:
    if hasattr(env, "STATE_DIM"):
        return int(env.STATE_DIM)
    # Fallback: reset, sniff shape, reset again to be safe.
    s = env.reset()
    dim = int(np.asarray(s).shape[-1])
    return dim


def _linear_schedule(step: int, start: float, end: float, n_steps: int) -> float:
    if step >= n_steps:
        return end
    frac = step / max(n_steps, 1)
    return start + (end - start) * frac
