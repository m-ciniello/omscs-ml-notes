import numpy as np


class GridWorld:
    """Configurable grid world environment with optional stochasticity.

    A 2D grid MDP.  The agent starts at `start`, must reach `goal`, and
    should avoid traps.  Supports walls (impassable cells), action slip
    (random action with some probability), and stochastic traps (penalty
    fires with some probability, normal step cost otherwise).

    All stochasticity is **stationary** — T and R are fixed functions of
    state, so VI/PI remain valid.  Learning agents handle stochasticity
    naturally through experience.

    Args:
        rows, cols: grid dimensions.
        goal: (row, col) of the goal cell.
        traps: set of (row, col) deterministic trap cells (always penalize).
        stochastic_traps: dict mapping (row, col) -> (penalty, probability).
            E.g., {(3,5): (-10, 0.4)} means 40% chance of -10, otherwise
            normal step cost.  Takes precedence over `traps` for that cell.
        walls: set of (row, col) impassable cells.  Moving into a wall
            bounces back (agent stays in place), like hitting the grid edge.
        slip_prob: probability that the intended action is replaced by a
            uniformly random action.  0.0 = deterministic, 0.2 = 20% slip.
        start: (row, col) where each episode begins.
        goal_reward, trap_reward, step_cost: reward values.
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTION_NAMES = ["up", "down", "left", "right"]
    N_ACTIONS = 4

    def __init__(
        self,
        rows=6,
        cols=6,
        goal=(0, 5),
        traps=None,
        stochastic_traps=None,
        walls=None,
        slip_prob=0.0,
        start=(5, 0),
        goal_reward=10.0,
        trap_reward=-10.0,
        step_cost=-0.1,
    ):
        self.rows = rows
        self.cols = cols
        self.goal = goal
        self.traps = set(traps) if traps else set()
        self.stochastic_traps = dict(stochastic_traps) if stochastic_traps else {}
        self.walls = set(walls) if walls else set()
        self.slip_prob = slip_prob
        self.start = start
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.step_cost = step_cost

    # ------------------------------------------------------------------
    # Core MDP interface
    # ------------------------------------------------------------------

    def _move(self, s, a):
        """Compute next state for action a from state s (no randomness).

        Handles wall collisions and grid-edge clipping.  Returns the
        next (row, col) — the reward is computed separately because it
        may be stochastic.
        """
        nr, nc = s[0] + self.ACTIONS[a][0], s[1] + self.ACTIONS[a][1]
        nr = max(0, min(self.rows - 1, nr))
        nc = max(0, min(self.cols - 1, nc))
        s2 = (nr, nc)
        if s2 in self.walls:
            return s  # bounce back
        return s2

    def _reward(self, s2):
        """Deterministic reward for landing in cell s2.

        For stochastic traps, this returns the *penalty* value.
        The probability is handled by step() and get_all_transitions().
        """
        if s2 == self.goal:
            return self.goal_reward
        if s2 in self.traps:
            return self.trap_reward
        if s2 in self.stochastic_traps:
            return self.stochastic_traps[s2][0]  # penalty value
        return self.step_cost

    def step(self, s, a, verbose=False):
        """Take action a from state s.  Returns (next_state, reward).

        If slip_prob > 0, the action is replaced by a random action with
        that probability.  If the destination is a stochastic trap, the
        penalty fires with the trap's probability.
        """
        # Action slip: with slip_prob, replace intended action with random
        if self.slip_prob > 0 and np.random.random() < self.slip_prob:
            a = np.random.randint(self.N_ACTIONS)

        s2 = self._move(s, a)

        # Compute reward (with stochastic trap roll)
        if s2 == self.goal:
            r = self.goal_reward
        elif s2 in self.traps:
            r = self.trap_reward
        elif s2 in self.stochastic_traps:
            penalty, prob = self.stochastic_traps[s2]
            r = penalty if np.random.random() < prob else self.step_cost
        else:
            r = self.step_cost

        if verbose:
            label = self.cell_label(s2).strip()
            tag = f" [{label}]" if label else ""
            print(f"  {s} --{self.ACTION_NAMES[a]}--> {s2}{tag}  r={r:+.1f}")
        return s2, r

    # ------------------------------------------------------------------
    # Methods for dynamic programming (VI / PI)
    # ------------------------------------------------------------------

    def all_states(self):
        """Return a list of every non-wall (row, col) state."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in self.walls]

    def get_all_transitions(self):
        """Pre-compute the full transition model.

        Returns a dict:
            {(s, a): [(prob, s', r), ...]}

        Each entry is a list of (probability, next_state, reward) tuples
        that sum to probability 1.0.  This is the same format Gymnasium
        uses for env.P, and it's what VI/PI need to compute expectations.

        Sources of branching:
        - slip_prob: intended action fires with (1 - slip_prob), each of
          the 4 actions fires with slip_prob/4.
        - stochastic_traps: landing on one produces two outcomes (penalty
          with trap_prob, step_cost with 1 - trap_prob).
        """
        transitions = {}
        for s in self.all_states():
            for a in range(self.N_ACTIONS):
                outcomes = {}  # (s', r) -> accumulated probability

                # Build action distribution: intended + possible slips
                if self.slip_prob > 0:
                    action_probs = []
                    for a2 in range(self.N_ACTIONS):
                        p = self.slip_prob / self.N_ACTIONS
                        if a2 == a:
                            p += (1 - self.slip_prob)
                        action_probs.append((a2, p))
                else:
                    action_probs = [(a, 1.0)]

                for a2, a_prob in action_probs:
                    s2 = self._move(s, a2)

                    if s2 in self.stochastic_traps:
                        penalty, trap_prob = self.stochastic_traps[s2]
                        # Outcome 1: trap fires
                        key1 = (s2, penalty)
                        outcomes[key1] = outcomes.get(key1, 0.0) + a_prob * trap_prob
                        # Outcome 2: trap doesn't fire
                        key2 = (s2, self.step_cost)
                        outcomes[key2] = outcomes.get(key2, 0.0) + a_prob * (1 - trap_prob)
                    else:
                        r = self._reward(s2)
                        key = (s2, r)
                        outcomes[key] = outcomes.get(key, 0.0) + a_prob

                transitions[(s, a)] = [
                    (prob, s2, r) for (s2, r), prob in outcomes.items()
                ]
        return transitions

    def is_terminal(self, s):
        """Return True if s is the goal (episode ends on reaching it)."""
        return s == self.goal

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def cell_label(self, s):
        if s == self.goal:
            return "GOAL"
        if s in self.walls:
            return "WALL"
        if s in self.traps:
            return "TRAP"
        if s in self.stochastic_traps:
            _, prob = self.stochastic_traps[s]
            return f"~{int(prob*100)}%"
        return ""

    def show_grid(self):
        """Print the grid layout: S=start, G=goal, X=trap, ~=stochastic trap,
        #=wall, .=empty."""
        for r in range(self.rows):
            row = ""
            for c in range(self.cols):
                cell = (r, c)
                if cell == self.start:
                    row += "  S "
                elif cell == self.goal:
                    row += "  G "
                elif cell in self.walls:
                    row += "  # "
                elif cell in self.traps:
                    row += "  X "
                elif cell in self.stochastic_traps:
                    row += "  ~ "
                else:
                    row += "  . "
            print(row)
        print()

    def __repr__(self):
        extras = []
        if self.walls:
            extras.append(f"walls={len(self.walls)}")
        if self.slip_prob > 0:
            extras.append(f"slip={self.slip_prob}")
        if self.stochastic_traps:
            extras.append(f"stoch_traps={len(self.stochastic_traps)}")
        extra_str = ", " + ", ".join(extras) if extras else ""
        return (
            f"GridWorld({self.rows}x{self.cols}, start={self.start}, "
            f"goal={self.goal}, traps={len(self.traps)}{extra_str})"
        )
