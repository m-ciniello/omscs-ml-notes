"""Deep Q-Network (DQN) agent with optional Double DQN.

DQN replaces the Q-table with a neural network: instead of storing one
number per (state, action) pair, we train a function Q_θ(s) → R^|A|
that takes a state and outputs Q-values for *all* actions at once.

This matters when the state space is too large to enumerate (e.g.,
continuous observations in CartPole, or pixel inputs in Atari).  On our
tiny GridWorld it's overkill — but that's the point: we can verify the
network learns the same policy as tabular Q-learning and VI.

Three ingredients distinguish DQN from tabular Q-learning:

1. **Neural network** (Q_θ):  Generalizes across states — updating one
   state's Q-values also shifts nearby states' estimates.  This is both
   the power (generalization) and the danger (catastrophic interference).

2. **Experience replay buffer**:  Stores past (s, a, r, s', done) tuples
   and samples random mini-batches for training.  Without this, the
   network sees correlated sequences of states and overfits to the
   current region of the state space.

3. **Target network** (Q_θ⁻):  A frozen copy of Q_θ, updated every
   C steps.  The TD target y = r + γ max_a' Q_θ⁻(s', a') stays fixed
   between updates, preventing the "chasing a moving target" instability.

**Double DQN** (optional, `double=True`):
   Vanilla DQN picks the max action AND evaluates it with the same
   (target) network, which causes systematic overestimation.  Double DQN
   decouples these: the *online* network selects the best action, but the
   *target* network evaluates it.  This reduces maximization bias and
   typically improves stability.

   Vanilla:   y = r + γ Q_θ⁻(s', argmax_a' Q_θ⁻(s', a'))
   Double:    y = r + γ Q_θ⁻(s', argmax_a' Q_θ(s', a'))
                                              ^^^^ online network picks
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Neural network: state → Q-values for all actions
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Small feedforward network: state (row, col) → Q(a) for each action.

    We normalize the (row, col) input to [0, 1] so the network sees
    similar-scale features regardless of grid size.
    """

    def __init__(self, state_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-size FIFO buffer that stores transitions and samples batches.

    Why not just train on each transition as it arrives?  Because
    consecutive transitions are highly correlated (state t+1 is similar
    to state t), and SGD assumes i.i.d. samples.  Storing transitions
    and sampling random batches breaks that correlation.
    """

    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """DQN agent for GridWorld, with optional Double DQN.

    Mirrors the TabularAgent interface: run_episodes(n), show(), etc.

    Args:
        env: GridWorld environment.
        double: if True, use Double DQN for the TD target.
        hidden: hidden layer size for the Q-network.
        lr: learning rate for Adam optimizer.
        gamma: discount factor.
        epsilon_start: initial exploration rate.
        epsilon_end: final exploration rate after decay.
        epsilon_decay: number of steps over which epsilon linearly decays.
        buffer_size: replay buffer capacity.
        batch_size: mini-batch size for training.
        target_update: copy online → target network every this many steps.
        min_replay: minimum buffer size before training starts.
        seed: random seed.
    """

    def __init__(self, env, double=False, hidden=64, lr=1e-3,
                 gamma=0.95, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=2000, buffer_size=10_000, batch_size=32,
                 target_update=100, min_replay=64, seed=0):
        self._seed_everything(seed)
        self.env = env
        self.double = double
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.min_replay = min_replay

        self.state_dim = 2  # (row, col) normalized
        self.n_actions = env.N_ACTIONS

        # Online network (the one we train) and target network (frozen copy)
        self.q_net = QNetwork(self.state_dim, self.n_actions, hidden)
        self.target_net = QNetwork(self.state_dim, self.n_actions, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay = ReplayBuffer(buffer_size)

        # Normalization constants for state encoding
        self._row_scale = max(env.rows - 1, 1)
        self._col_scale = max(env.cols - 1, 1)

        self.s = env.start
        self.total_steps = 0
        self.episode = 0
        self.episode_steps = 0
        self.train_losses = []

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @property
    def name(self):
        return "Double DQN" if self.double else "DQN"

    # --- State encoding ---

    def _encode(self, s):
        """Convert (row, col) tuple to a normalized float tensor.

        Normalization to [0, 1] helps the network learn faster — without
        it, gradients would be dominated by whichever coordinate is larger.
        """
        return torch.FloatTensor([s[0] / self._row_scale,
                                  s[1] / self._col_scale])

    def _encode_batch(self, states):
        """Encode a batch of states into a (batch, 2) tensor."""
        return torch.FloatTensor(
            [[s[0] / self._row_scale, s[1] / self._col_scale]
             for s in states])

    # --- Action selection ---

    def _select_action(self, s):
        """ε-greedy: explore with probability ε, exploit otherwise.

        ε decays linearly from epsilon_start to epsilon_end over
        epsilon_decay steps — high exploration early, mostly greedy later.
        """
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - self.total_steps / self.epsilon_decay
                * (self.epsilon_start - self.epsilon_end))

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            q_vals = self.q_net(self._encode(s))
            return q_vals.argmax().item()

    # --- Training step ---

    def _train_step(self):
        """Sample a mini-batch from replay and do one gradient step.

        The loss is MSE between predicted Q(s, a) and the TD target:
            y = r + γ max_a' Q_target(s', a')              (vanilla DQN)
            y = r + γ Q_target(s', argmax_a' Q_online(s'))  (Double DQN)
        """
        if len(self.replay) < self.min_replay:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size)

        s_t = self._encode_batch(states)
        s2_t = self._encode_batch(next_states)
        a_t = torch.LongTensor(actions)
        r_t = torch.FloatTensor(rewards)
        done_t = torch.FloatTensor(dones)

        # Q(s, a) for the actions actually taken — gather picks one Q-value
        # per row using the action index
        q_values = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

        # TD target: what we "observed" (one-step lookahead)
        with torch.no_grad():
            if self.double:
                # Double DQN: online network picks the action,
                # target network evaluates it.
                # This prevents the overestimation that happens when the
                # same network both selects AND evaluates the max action.
                best_actions = self.q_net(s2_t).argmax(dim=1)
                next_q = self.target_net(s2_t).gather(
                    1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                # Vanilla DQN: target network does both selection and evaluation
                next_q = self.target_net(s2_t).max(dim=1).values

            # If the episode ended (done=1), there are no future rewards
            td_target = r_t + self.gamma * next_q * (1 - done_t)

        loss = self.loss_fn(q_values, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_losses.append(loss.item())

    # --- Core step ---

    def single_step(self, verbose=False):
        """Take one step, store in replay, train, maybe update target.
        Returns True if episode ended."""
        s = self.s
        a = self._select_action(s)
        s2, r = self.env.step(s, a, verbose=verbose)
        done = self.env.is_terminal(s2)

        self.replay.push(s, a, r, s2, float(done))
        self._train_step()

        self.total_steps += 1
        self.episode_steps += 1

        # Periodically copy online weights → target network
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if verbose:
            with torch.no_grad():
                q_vals = self.q_net(self._encode(s))
            print(f"    Q({s},{self.env.ACTION_NAMES[a]}): {q_vals[a].item():.3f}  "
                  f"ε={self.epsilon:.3f}")

        self.s = s2
        if done:
            if verbose:
                print(f"  *** GOAL in {self.episode_steps} steps "
                      f"(episode {self.episode}) ***")
            self.episode += 1
            self.episode_steps = 0
            self.s = self.env.start

        return done

    def run_episode(self, verbose=False, max_steps=200):
        """Run one full episode. Returns number of steps taken."""
        for t in range(max_steps):
            if self.single_step(verbose=verbose):
                return t + 1
        return max_steps

    def run_episodes(self, n, verbose=False, max_steps=200):
        """Run n episodes. Returns list of step counts per episode."""
        return [self.run_episode(verbose=verbose, max_steps=max_steps)
                for _ in range(n)]

    # --- Q-table extraction (for display) ---

    @property
    def Q(self):
        """Extract a Q-table from the network for visualization.

        Returns an (rows, cols, N_ACTIONS) array — same shape as
        TabularAgent.Q — so our show_policy_and_values function works.
        """
        q_table = np.zeros((self.env.rows, self.env.cols, self.n_actions))
        with torch.no_grad():
            for r in range(self.env.rows):
                for c in range(self.env.cols):
                    s = (r, c)
                    q_vals = self.q_net(self._encode(s))
                    q_table[r, c] = q_vals.numpy()
        return q_table

    def show(self):
        """Print current policy and value grid (extracted from network)."""
        from .tabular import show_policy_and_values
        show_policy_and_values(self.Q, self.env)
        print(f"[{self.name}]  Total steps: {self.total_steps}  |  "
              f"Episodes: {self.episode}  |  "
              f"Replay buffer: {len(self.replay)}  |  "
              f"ε: {self.epsilon:.3f}")
