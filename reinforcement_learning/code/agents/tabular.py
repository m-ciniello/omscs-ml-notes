"""Unified tabular RL agent: Q-Learning, SARSA, Dyna-Q, and Prioritized Sweeping.

All four are variations of the same core loop — take a step, compute a TD error,
update Q(s, a).  The differences reduce to three flags:

- **on_policy** (False = Q-learning, True = SARSA):
    Q-learning: td_target = r + γ max_a' Q(s', a')    "best I COULD do from s'"
    SARSA:      td_target = r + γ Q(s', a')            "what I'll ACTUALLY do from s'"
    One line of code differs.  Q-learning converges to Q* (optimal); SARSA converges
    to the value of the ε-greedy policy (safer, accounts for exploration noise).

- **k_sim** (0 = model-free, >0 = model-based):
    With k_sim=0, each real transition updates Q once and is discarded.
    With k_sim>0, each transition is stored in a model dict and replayed k times
    per real step — same math, using the model instead of the real environment.
    More replay → faster convergence, at the cost of extra computation.

- **prioritized** (only matters when k_sim > 0):
    False = random replay (Dyna): pick k random past transitions to replay.
    True = prioritized sweeping: replay the transitions with the largest TD errors
    first, cascading backward through predecessors.
"""

import numpy as np


def show_policy_and_values(Q, env):
    """Display the learned policy (arrows) and max_a Q(s,a) side by side."""
    arrows = ["↑", "↓", "←", "→"]
    pw = env.cols * 6
    vw = env.cols * 6
    print(f"{'Policy':^{pw}s}    {'V*(s) = max_a Q(s,a)':^{vw}s}")
    print(f"{'-' * pw:s}    {'-' * vw:s}")
    for r in range(env.rows):
        policy_row = ""
        value_row = ""
        for c in range(env.cols):
            if (r, c) == env.goal:
                policy_row += "  G   "
                value_row += "  G   "
            elif (r, c) in env.walls:
                policy_row += "  #   "
                value_row += "  #   "
            else:
                best_a = np.argmax(Q[r, c])
                policy_row += f"  {arrows[best_a]}   "
                value_row += f"{np.max(Q[r, c]):5.2f} "
        print(f"|{policy_row}|    |{value_row}|")
    print()


class TabularAgent:
    """Unified tabular RL agent.

    Covers Q-learning, SARSA, Dyna-Q, and Prioritized Sweeping via flags:
        TabularAgent(env)                              → Q-learning
        TabularAgent(env, on_policy=True)              → SARSA
        TabularAgent(env, k_sim=50)                    → Dyna-Q
        TabularAgent(env, k_sim=50, prioritized=True)  → Prioritized Sweeping

    Args:
        env: GridWorld environment.
        on_policy: if True, use SARSA (actual next action) instead of
            Q-learning (max over next actions) for the TD target.
        k_sim: number of simulated replay updates per real step.
            0 = model-free (no replay), >0 = model-based.
        prioritized: if True and k_sim > 0, use prioritized sweeping
            instead of random replay.
        priority_theta: minimum TD error to enter the priority queue.
        epsilon: probability of taking a random action (exploration).
        gamma: discount factor.
        alpha: learning rate.
        seed: random seed for reproducibility.
    """

    def __init__(self, env, on_policy=False, k_sim=0, prioritized=False,
                 priority_theta=1e-4, epsilon=0.1, gamma=0.95, alpha=0.1,
                 seed=0):
        np.random.seed(seed)
        self.env = env
        self.on_policy = on_policy
        self.k_sim = k_sim
        self.prioritized = prioritized
        self.priority_theta = priority_theta
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.Q = np.zeros((env.rows, env.cols, env.N_ACTIONS))

        # The "model" is a lookup table of observed transitions:
        # (s, a) -> (s', r).  Only used when k_sim > 0.
        self.model = {}
        self.visited = []
        self.predecessors = {}  # s' -> set of (s, a) that lead to s'
        # Priority queue: (s, a) -> TD error magnitude.
        # Highest error = updated first.  Only used when prioritized=True.
        self.pq = {}

        self.s = env.start
        self.a = self._select_action(self.s)  # needed for SARSA; harmless otherwise
        self.total_steps = 0
        self.episode = 0
        self.episode_steps = 0

    @property
    def name(self):
        """Human-readable name for the current configuration."""
        if self.k_sim == 0:
            return "SARSA" if self.on_policy else "Q-learning"
        mode = "PriSweep" if self.prioritized else "Dyna"
        policy = "+SARSA" if self.on_policy else ""
        return f"{mode}(k={self.k_sim}){policy}"

    def _select_action(self, s):
        """ε-greedy action selection."""
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[s])
        return np.random.randint(self.env.N_ACTIONS)

    # --- Priority queue helpers (only used when prioritized=True) ---

    def _pq_push(self, s, a, priority):
        """Insert/update (s, a) in priority queue if above threshold."""
        if priority > self.priority_theta:
            key = (s, a)
            if key not in self.pq or priority > self.pq[key]:
                self.pq[key] = priority

    def _pq_pop(self):
        """Pop the (s, a) pair with highest priority. Returns None if empty."""
        if not self.pq:
            return None
        key = max(self.pq, key=self.pq.get)
        del self.pq[key]
        return key

    def _promote_predecessors(self, s):
        """After updating a state, check if its predecessors need requeueing."""
        if s not in self.predecessors:
            return
        for (sp, ap) in self.predecessors[s]:
            s2p, rp = self.model[(sp, ap)]
            p = abs(rp + self.gamma * np.max(self.Q[s2p]) - self.Q[sp][ap])
            self._pq_push(sp, ap, p)

    # --- Core learning step ---

    def single_step(self, verbose=False):
        """Take one real action, update Q, optionally run simulated updates.
        Returns True if the episode ended."""
        s = self.s
        a = self.a if self.on_policy else self._select_action(s)

        s2, r = self.env.step(s, a, verbose=verbose)

        # --- TD target: the ONE line that distinguishes Q-learning from SARSA ---
        if self.on_policy:
            a2 = self._select_action(s2)
            td_target = r + self.gamma * self.Q[s2][a2]
        else:
            a2 = None
            td_target = r + self.gamma * np.max(self.Q[s2])

        old_q = self.Q[s][a]
        self.Q[s][a] += self.alpha * (td_target - old_q)

        if verbose:
            target_desc = (f"Q(s',{self.env.ACTION_NAMES[a2]})" if self.on_policy
                           else "max Q(s')")
            print(f"    Q({s},{self.env.ACTION_NAMES[a]}): "
                  f"{old_q:.3f} → {self.Q[s][a]:.3f}  "
                  f"(td_error={td_target - old_q:+.3f}, target via {target_desc})")

        # --- Model update (only when k_sim > 0) ---
        if self.k_sim > 0:
            self.model[(s, a)] = (s2, r)
            if (s, a) not in self.visited:
                self.visited.append((s, a))
            if s2 not in self.predecessors:
                self.predecessors[s2] = set()
            self.predecessors[s2].add((s, a))

            # --- Simulated replay ---
            if self.prioritized:
                self._pq_push(s, a, abs(td_target - old_q))
                self._promote_predecessors(s)

                n_sim = 0
                for i in range(self.k_sim):
                    pair = self._pq_pop()
                    if pair is None:
                        break
                    si, ai = pair
                    s2i, ri = self.model[(si, ai)]
                    old_qi = self.Q[si][ai]
                    self.Q[si][ai] += self.alpha * (
                        ri + self.gamma * np.max(self.Q[s2i]) - old_qi)
                    n_sim += 1
                    if verbose and self.k_sim <= 10:
                        print(f"      pri {i+1}: Q({si},{self.env.ACTION_NAMES[ai]}): "
                              f"{old_qi:.3f} → {self.Q[si][ai]:.3f}")
                    self._promote_predecessors(si)
                if verbose and n_sim < self.k_sim:
                    print(f"    (queue emptied after {n_sim} updates)")
            else:
                for i in range(self.k_sim):
                    si, ai = self.visited[np.random.randint(len(self.visited))]
                    s2i, ri = self.model[(si, ai)]
                    old_qi = self.Q[si][ai]
                    self.Q[si][ai] += self.alpha * (
                        ri + self.gamma * np.max(self.Q[s2i]) - old_qi)
                    if verbose and self.k_sim <= 10:
                        print(f"      sim {i+1}: Q({si},{self.env.ACTION_NAMES[ai]}): "
                              f"{old_qi:.3f} → {self.Q[si][ai]:.3f}")

        # --- Advance state ---
        self.s = s2
        if self.on_policy:
            self.a = a2
        self.total_steps += 1
        self.episode_steps += 1

        done = self.env.is_terminal(s2)
        if done:
            if verbose:
                print(f"  *** GOAL in {self.episode_steps} steps "
                      f"(episode {self.episode}) ***")
            self.episode += 1
            self.episode_steps = 0
            self.s = self.env.start
            if self.on_policy:
                self.a = self._select_action(self.s)

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

    def show(self):
        """Print current policy, value grid, and agent stats."""
        show_policy_and_values(self.Q, self.env)
        info = (f"[{self.name}]  Total steps: {self.total_steps}  |  "
                f"Episodes: {self.episode}")
        if self.k_sim > 0:
            info += f"  |  Model entries: {len(self.model)}"
        print(info)
