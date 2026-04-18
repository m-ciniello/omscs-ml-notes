"""Dynamic Programming solvers: Value Iteration and Policy Iteration.

These require the full model (T, R) — they solve the MDP *without*
interacting with it.  They handle both deterministic and stochastic
environments: the Bellman equation becomes an expectation over
possible outcomes weighted by their probabilities.
"""

import numpy as np


def show_policy_and_values(V, policy, env):
    """Display the policy (arrows) and V(s) side by side."""
    arrows = ["↑", "↓", "←", "→"]
    pw = env.cols * 6
    vw = env.cols * 6
    print(f"{'Policy':^{pw}s}    {'V(s)':^{vw}s}")
    print(f"{'-' * pw:s}    {'-' * vw:s}")
    for r in range(env.rows):
        policy_row = ""
        value_row = ""
        for c in range(env.cols):
            s = (r, c)
            if s == env.goal:
                policy_row += "  G   "
                value_row += "  G   "
            elif s in env.walls:
                policy_row += "  #   "
                value_row += "  #   "
            elif s in env.traps:
                policy_row += "  X   "
                value_row += f"{V[r, c]:5.2f} "
            else:
                policy_row += f"  {arrows[policy[r, c]]}   "
                value_row += f"{V[r, c]:5.2f} "
        print(f"|{policy_row}|    |{value_row}|")
    print()


class _DPBase:
    """Shared scaffolding for DP solvers.

    Provides the common state (env, gamma, V, policy, transitions)
    and two helpers that both VI and PI need:
      - _greedy_action: the argmax-over-actions loop
      - run_episode / run_episodes: run the solved policy in the env
        (matches the API of TabularAgent and DQNAgent so all agents
        share one benchmarking entry point).
    """

    def __init__(self, env, gamma=0.95):
        self.env = env
        self.gamma = gamma
        self.transitions = env.get_all_transitions()
        self.V = np.zeros((env.rows, env.cols))
        self.policy = np.zeros((env.rows, env.cols), dtype=int)
        self.iterations = 0

    def _greedy_action(self, s):
        """Return (argmax_a Q(s, a), max_a Q(s, a)) under the current V.

        Q(s, a) = Σ p * (r + γ V(s')) over outcomes (p, s', r) — the
        Bellman equation with the expectation made explicit.  See
        rl_foundations.md §2.4 (Q-function) and the §2.1 aside on why
        the per-transition reward form matches R(s, a) in expressiveness.

        Used by VI's Bellman update, VI's policy extraction, and PI's
        policy improvement — factored here so the argmax is written once.
        """
        q_values = np.array([
            sum(p * (r + self.gamma * self.V[s2])
                for p, s2, r in self.transitions[(s, a)])
            for a in range(self.env.N_ACTIONS)
        ])
        return int(np.argmax(q_values)), float(q_values.max())

    def _extract_policy(self):
        """Set π(s) = argmax_a [R(s,a) + γ V(s')] for every state."""
        for s in self.env.all_states():
            if self.env.is_terminal(s):
                continue
            self.policy[s], _ = self._greedy_action(s)

    def show(self):
        show_policy_and_values(self.V, self.policy, self.env)

    def run_episode(self, verbose=False, max_steps=200):
        """Run one episode of the solved policy. Returns number of steps.

        Matches the signature of TabularAgent.run_episode and
        DQNAgent.run_episode so all agents share one benchmarking API.
        """
        s = self.env.start
        for t in range(max_steps):
            a = self.policy[s]
            s_next, r = self.env.step(s, a)
            if verbose:
                print(f"  t={t}: s={s} a={a} r={r:+.2f} s'={s_next}")
            s = s_next
            if self.env.is_terminal(s):
                return t + 1
        return max_steps

    def run_episodes(self, n, verbose=False, max_steps=200):
        """Run n episodes of the solved policy. Returns list of step counts."""
        return [self.run_episode(verbose=verbose, max_steps=max_steps)
                for _ in range(n)]


class ValueIteration(_DPBase):
    """Find V* and π* by repeatedly applying the Bellman optimality equation.

    Algorithm (from the companion notes, Section 3.1):
        Start with V(s) = 0 for all s.
        Repeat until the largest change in V is below theta:
            For each state s:
                V(s) = max_a [ R(s,a) + gamma * V(s') ]
        Extract the greedy policy: π(s) = argmax_a [ R(s,a) + gamma * V(s') ]

    This works because the Bellman optimality operator is a contraction
    mapping with factor gamma — each sweep brings V closer to V* by at
    least a factor of gamma (Banach fixed-point theorem).
    """

    def __init__(self, env, gamma=0.95):
        super().__init__(env, gamma)
        self.history = []  # max Bellman residual per iteration

    def solve(self, theta=1e-6, max_iter=1000):
        """Run value iteration until convergence.

        Args:
            theta: stop when max|V_new - V_old| < theta (Bellman residual).
            max_iter: safety cap on iterations.

        Returns:
            Number of iterations to convergence.
        """
        for i in range(max_iter):
            delta = 0.0
            for s in self.env.all_states():
                if self.env.is_terminal(s):
                    continue
                old_v = self.V[s]
                _, best_v = self._greedy_action(s)
                self.V[s] = best_v
                delta = max(delta, abs(old_v - best_v))

            self.history.append(delta)
            self.iterations = i + 1

            if delta < theta:
                break

        self._extract_policy()
        return self.iterations

    def show(self):
        super().show()
        print(f"Converged in {self.iterations} iterations  |  gamma={self.gamma}")


class PolicyIteration(_DPBase):
    """Find V* and π* by alternating policy evaluation and improvement.

    Algorithm (from the companion notes, Section 3.2):
        1. Start with an arbitrary policy π (e.g., always go up).
        2. Policy evaluation: compute V^π by solving the Bellman equation
           for the current policy (iteratively until convergence).
        3. Policy improvement: update π(s) = argmax_a [R(s,a) + γ V(s')]
           for every state.
        4. If π didn't change, stop — we've found the optimal policy.
           Otherwise, go back to step 2.

    PI typically converges in far fewer outer iterations than VI, but
    each iteration is more expensive because policy evaluation must
    itself iterate to convergence.
    """

    def __init__(self, env, gamma=0.95):
        super().__init__(env, gamma)
        self.eval_iterations = []  # per-round evaluation iteration counts

    def solve(self, eval_theta=1e-6, max_iter=100):
        """Run policy iteration until the policy stabilizes.

        Returns:
            Number of policy improvement rounds.
        """
        for i in range(max_iter):
            eval_iters = self._evaluate_policy(eval_theta)
            self.eval_iterations.append(eval_iters)

            policy_stable = self._improve_policy()

            self.iterations = i + 1
            if policy_stable:
                break

        return self.iterations

    def _evaluate_policy(self, theta, max_iter=1000):
        """Compute V^π for the current policy by iterating the Bellman
        equation for π (no max — just follow the policy).

        V^π(s) = E[R + γ V^π(s')] under action π(s)
               = Σ p * (r + γ V(s'))  for all outcomes of (s, π(s))
        """
        for i in range(max_iter):
            delta = 0.0
            for s in self.env.all_states():
                if self.env.is_terminal(s):
                    continue
                old_v = self.V[s]
                a = self.policy[s]
                self.V[s] = sum(
                    p * (r + self.gamma * self.V[s2])
                    for p, s2, r in self.transitions[(s, a)])
                delta = max(delta, abs(old_v - self.V[s]))
            if delta < theta:
                return i + 1
        return max_iter

    def _improve_policy(self):
        """Update π(s) = argmax_a [R(s,a) + γ V(s')] for every state.

        Returns True if the policy didn't change (converged).
        """
        stable = True
        for s in self.env.all_states():
            if self.env.is_terminal(s):
                continue
            old_a = self.policy[s]
            self.policy[s], _ = self._greedy_action(s)
            if old_a != self.policy[s]:
                stable = False
        return stable

    def show(self):
        super().show()
        print(
            f"Converged in {self.iterations} policy improvement rounds  |  "
            f"gamma={self.gamma}"
        )
        if self.eval_iterations:
            print(
                f"Evaluation iterations per round: {self.eval_iterations}"
            )
