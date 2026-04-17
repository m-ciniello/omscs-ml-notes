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
      - evaluate_policy: run the solved policy on the live environment
    """

    def __init__(self, env, gamma=0.95):
        self.env = env
        self.gamma = gamma
        self.transitions = env.get_all_transitions()
        self.V = np.zeros((env.rows, env.cols))
        self.policy = np.zeros((env.rows, env.cols), dtype=int)
        self.iterations = 0

    def _q_value(self, s, a):
        """Compute Q(s, a) = E[R + γ V(s')] under the transition model.

        With stochastic transitions, each (s, a) can produce multiple
        outcomes.  We take the probability-weighted sum:
            Q(s, a) = Σ p * (r + γ V(s'))
        This is the Bellman equation with the expectation made explicit.
        """
        return sum(p * (r + self.gamma * self.V[s2])
                   for p, s2, r in self.transitions[(s, a)])

    def _greedy_action(self, s):
        """Return (best_action, best_value) for state s under current V.

        This is the argmax_a Q(s, a) loop that appears in VI's Bellman
        update, VI's policy extraction, and PI's policy improvement —
        factored out here so it's written once.
        """
        best_a, best_v = 0, -np.inf
        for a in range(self.env.N_ACTIONS):
            q_sa = self._q_value(s, a)
            if q_sa > best_v:
                best_a, best_v = a, q_sa
        return best_a, best_v

    def _extract_policy(self):
        """Set π(s) = argmax_a [R(s,a) + γ V(s')] for every state."""
        for s in self.env.all_states():
            if self.env.is_terminal(s):
                continue
            self.policy[s], _ = self._greedy_action(s)

    def show(self):
        show_policy_and_values(self.V, self.policy, self.env)

    def evaluate_policy(self, n_episodes=100, max_steps=200):
        """Run the solved policy on the environment and return step counts.

        Useful for comparing against learning agents on the same metric
        (steps-to-goal per episode).
        """
        step_counts = []
        for _ in range(n_episodes):
            s = self.env.start
            for t in range(max_steps):
                a = self.policy[s]
                s, _ = self.env.step(s, a)
                if self.env.is_terminal(s):
                    step_counts.append(t + 1)
                    break
            else:
                step_counts.append(max_steps)
        return step_counts


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
