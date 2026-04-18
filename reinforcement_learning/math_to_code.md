# From Math to Code: Implementation Notes for the RL Algorithms

This document bridges the theory notes (`rl_foundations.md`, `rl_in_practice.md`, `policy_gradients.md`) and the implementations in `code/agents/`. Each section is a short, focused note answering a question of the form *"the theory looks like X but the code looks like Y — why?"*

The goal is to capture the design choices, equivalences, and little re-writes that make the code simpler or cleaner than a literal transcription of the math would be. Things that are too concrete for the theory notes and too verbose for a docstring live here.

Organized by source file. Cross-references use `rl_foundations.md §2.4` for theory and `dp.py::ValueIteration.solve` for code.

---

## Table of Contents

- [1. Dynamic Programming — `code/agents/dp.py`](#1-dynamic-programming--codeagentsdppy)
  - [1.1 The reward form: `(p, s', r)` tuples vs. $R(s, a)$](#11-the-reward-form-p-s-r-tuples-vs-rs-a)
  - [1.2 In-place updates: Gauss-Seidel, not Jacobi](#12-in-place-updates-gauss-seidel-not-jacobi)
  - [1.3 Where does the policy live during VI's solve loop?](#13-where-does-the-policy-live-during-vis-solve-loop)
  - [1.4 Policy iteration: iterative vs. direct policy evaluation](#14-policy-iteration-iterative-vs-direct-policy-evaluation)
  - [1.5 Why `_q_value` and `_greedy_action` are merged](#15-why-_q_value-and-_greedy_action-are-merged)
  - [1.6 The `run_episode` / `run_episodes` API (shared across all agents)](#16-the-run_episode--run_episodes-api-shared-across-all-agents)
  - [1.7 VI vs. PI in practice: what the two solvers feel like when you run them](#17-vi-vs-pi-in-practice-what-the-two-solvers-feel-like-when-you-run-them)
- [2. Deep Q-Networks — `code/agents/dqn.py`](#2-deep-q-networks--codeagentsdqnpy)
  - [2.1 Why the state is normalized before going into `QNetwork`](#21-why-the-state-is-normalized-before-going-into-qnetwork)
  - [2.2 The replay buffer: what it does, why it works, and where it wouldn't](#22-the-replay-buffer-what-it-does-why-it-works-and-where-it-wouldnt)
  - [2.3 Action selection: ε-greedy decay and `torch.no_grad()`](#23-action-selection-ε-greedy-decay-and-torchno_grad)
  - [2.4 The training step: `gather`, semi-gradients, and Double DQN](#24-the-training-step-gather-semi-gradients-and-double-dqn)
  - [2.5 Target network updates: the cadence knob](#25-target-network-updates-the-cadence-knob)

---

## 1. Dynamic Programming — `code/agents/dp.py`

Theory: `rl_foundations.md §3.1` (Value Iteration) and `§3.2` (Policy Iteration).

### 1.1 The reward form: `(p, s', r)` tuples vs. $R(s, a)$

The notes write the Bellman optimality equation for $Q^*$ as

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} T(s, a, s') V^*(s').$$

The code stores each transition as a tuple `(p, s2, r)` and computes

```python
sum(p * (r + self.gamma * self.V[s2]) for p, s2, r in self.transitions[(s, a)])
```

These look different — the notes pull $R(s, a)$ *outside* the sum, the code keeps a per-outcome `r` *inside* it. Both compute the same value, because of the term-by-term identity

$$R(s,a) + \gamma \sum_{s'} T(s,a,s') V(s') \;=\; \sum_{s'} T(s,a,s')\big[R(s,a,s') + \gamma V(s')\big]$$

valid whenever $R(s, a, s') = R(s, a)$ for all $s'$ (the "reward is a property of the action, not the landing state" case).

The code uses the per-outcome form because it is strictly more general — $R(s, a, s')$ can encode "reward depends on the landing cell" (goal vs. step penalty in a grid world), which is awkward to write as $R(s, a)$. The extra generality is free: the sum absorbs it either way. See the aside in `rl_foundations.md §2.1` for the full equivalence argument.

**Takeaway:** if you're reading the code and see `r` inside the `sum(...)`, mentally group it with the $\sum T V$ term rather than pulling it out. The math works either way.

### 1.2 In-place updates: Gauss-Seidel, not Jacobi

The theory box for value iteration in `rl_foundations.md §3.1` reads:

> For each state $s$: $V(s) := \max_a [R(s, a) + \gamma \sum_{s'} T(s, a, s') V(s')]$

Read literally, this is ambiguous about *which* $V$ appears on the right-hand side: the $V$ from the start of this sweep, or the $V$ that has already been partially updated during this sweep?

The code makes a choice:

```python
for s in self.env.all_states():
    ...
    self.V[s] = best_v   # writes immediately, before the next state is visited
```

Later states in the same sweep read this new `V[s]`, not the pre-sweep value. That's **Gauss-Seidel** iteration (updates propagate within a sweep). The alternative — **Jacobi** — would copy `V` into `V_new`, write all updates to `V_new`, and swap at the end of the sweep, so every state in a sweep reads the pre-sweep values.

Both converge to the same $V^*$. Gauss-Seidel typically converges in fewer sweeps because information can propagate across states within a sweep rather than only across sweeps. The contraction argument from `rl_foundations.md §3.1` still applies (the Bellman optimality operator remains a $\gamma$-contraction under either update order), so correctness is preserved.

**Takeaway:** the `self.V[s] = best_v` inside the loop is not a bug waiting to happen — it's a deliberate choice that makes each sweep slightly more efficient. The order of iteration over states now affects *speed* but not *correctness*.

### 1.3 Where does the policy live during VI's solve loop?

Look at the VI solve loop carefully:

```python
for i in range(max_iter):
    delta = 0.0
    for s in self.env.all_states():
        ...
        _, best_v = self._greedy_action(s)   # ← the action is discarded!
        self.V[s] = best_v
        ...
    ...

self._extract_policy()   # ← policy shows up here, once, at the end
```

The `_` throws away the argmax action on every state, every sweep. Only the `max` value is kept. This is *not* an oversight: during the solve loop, VI doesn't need a policy, it needs $V^*$. The policy is a *derived* object, computed once at the end by one final sweep of $\pi(s) = \arg\max_a Q(s, a)$ under the converged $V^*$.

This is a real structural difference between VI and PI:

- **VI** maintains only $V$. The policy is derived at the end.
- **PI** maintains both $V$ and $\pi$ from the start. The policy is updated on every outer iteration.

Why does this matter conceptually? It clarifies what each algorithm is "really doing":

- VI is **solving the Bellman optimality equation for $V^*$**. The policy falls out as a consequence.
- PI is **alternating fixed-point problems**. Evaluate $V^\pi$, then improve $\pi$, then repeat. Each loop strictly improves the policy until a fixed point is reached.

Both converge to the same answer — the Bellman optimality equation has a unique solution — but they get there via different objects.

**Takeaway:** if you're reading the VI code and wondering why no policy appears until `_extract_policy()` at the very end, that's by design. The policy doesn't participate in the recursion.

### 1.4 Policy iteration: iterative vs. direct policy evaluation

The notes (`§3.2`) present policy evaluation two ways:

- **Directly**, as a system of $|S|$ linear equations in $|S|$ unknowns, solvable in $O(|S|^3)$ via Gaussian elimination.
- **Iteratively**, by repeatedly applying the Bellman backup for $\pi$ (no max — just follow the policy) until the value residual is small.

The code uses the iterative version:

```python
def _evaluate_policy(self, theta, max_iter=1000):
    for i in range(max_iter):
        delta = 0.0
        for s in self.env.all_states():
            ...
            a = self.policy[s]
            self.V[s] = sum(
                p * (r + self.gamma * self.V[s2])
                for p, s2, r in self.transitions[(s, a)])
            delta = max(delta, abs(old_v - self.V[s]))
        if delta < theta:
            return i + 1
    return max_iter
```

Structurally this is *almost identical* to VI's inner loop — the same `sum(p * (r + γ V))` formula, the same `delta` tracking, the same in-place Gauss-Seidel update. The only difference: **no max**. Instead of asking "what's the best action here?", it just follows `self.policy[s]`.

Why iterate instead of solving the linear system exactly?

1. **Sparsity.** For any reasonable environment, $T$ is sparse — each `(s, a)` has a handful of outcomes, not $|S|$. Iterative backups exploit this natively; Gaussian elimination would need sparse-matrix techniques to match.
2. **Consistency with VI.** Same code patterns, same stopping criterion, same $\gamma$-contraction convergence argument. The two algorithms share machinery.
3. **"Good enough" is free.** PI doesn't actually need $V^\pi$ to full precision — it just needs enough accuracy that the argmax in policy improvement picks the right action. Modified policy iteration (`§3.2` end note) takes this further and runs only a handful of backups per outer round.

**Takeaway:** PI's `_evaluate_policy` is basically "VI's inner loop without the max." That's not a coincidence — both are applying the Bellman operator; VI applies the optimality operator $\mathcal{T}$, PI applies the policy operator $\mathcal{T}^\pi$. Same contraction mechanism, different argmax behavior.

### 1.5 Why `_q_value` and `_greedy_action` are merged

Earlier versions of the code had two helpers:

```python
def _q_value(self, s, a):
    return sum(p * (r + γ V[s2]) for p, s2, r in transitions[(s, a)])

def _greedy_action(self, s):
    best_a, best_v = 0, -inf
    for a in range(N_ACTIONS):
        q_sa = self._q_value(s, a)
        if q_sa > best_v:
            best_a, best_v = a, q_sa
    return best_a, best_v
```

The current code folds them into one function with a list comprehension and `np.argmax`:

```python
def _greedy_action(self, s):
    q_values = np.array([
        sum(p * (r + self.gamma * self.V[s2])
            for p, s2, r in self.transitions[(s, a)])
        for a in range(self.env.N_ACTIONS)
    ])
    return int(np.argmax(q_values)), float(q_values.max())
```

Why merge? `_q_value` was only ever called from `_greedy_action`. (`_evaluate_policy` inlines the sum itself — see §1.4 — because it never needs the argmax.) Keeping it as a separate method suggested reusability that didn't exist and hid the natural pattern "compute all Q-values for this state, then take the argmax." Merging exposes that pattern.

The numpy form is also more honest about what the loop is doing: it builds a $|A|$-vector of Q-values and takes the argmax/max in one step, rather than threading "best so far" through a scalar loop.

**Takeaway:** factoring helpers is worthwhile when they appear in multiple places or carry meaningful names. When neither is true, inlining can make the structure clearer.

### 1.6 The `run_episode` / `run_episodes` API (shared across all agents)

*Note: this section documents a convention that spans all agent types, not just DP. It lives here because it's where we first hit it.*

All three agent types — `_DPBase` (inherited by `ValueIteration` and `PolicyIteration`), `TabularAgent`, and `DQNAgent` — expose the same two methods:

- `run_episode(verbose=False, max_steps=200) -> int` — step count for one episode
- `run_episodes(n, verbose=False, max_steps=200) -> list[int]` — step counts for n episodes

This uniform API is deliberate: it means notebook code like

```python
for agent in [vi, pi, q_learner, dqn]:
    steps = agent.run_episodes(100)
    print(f"mean steps = {np.mean(steps):.1f}")
```

works regardless of what kind of agent it is. Behind the scenes the three implementations do quite different things:

- **DP solvers** (`vi`, `pi`): the policy is already solved and cached in `self.policy`. `run_episode` greedy-selects the stored action at each state and steps the env. No learning happens.
- **Tabular learner**: `run_episode` calls `single_step` in a loop, which executes the Q-learning (or SARSA, or Dyna) update after each transition. The policy evolves during the episode.
- **DQN**: same structure as tabular, but `single_step` also includes a gradient step on the Q-network.

So the same method name has one meaning at the call site ("run the agent for an episode") and different internals depending on whether the agent is a solver or a learner.

**Naming clash we resolved.** The earlier version of `dp.py` had a public method named `evaluate_policy` for this rollout. That clashed with PI's private `_evaluate_policy`, which is the Bellman-iteration step that computes $V^\pi$ (theory notes §3.2). "Policy evaluation" is a term of art in the theory, so we kept `_evaluate_policy` for PI and renamed the rollout method to match the learning agents' convention. Now:

- `_evaluate_policy` (private, `PolicyIteration` only) → the $\mathcal{T}^\pi$-iteration from §3.2.
- `run_episode` / `run_episodes` (public, all agents) → empirical rollout against `env`.

**Takeaway:** when you see `agent.run_episode(...)` in a notebook, trust that it does the right thing for whatever agent you've got. The uniform entry point is what makes side-by-side comparisons (DP optimal vs. Q-learning vs. DQN) clean to write.

### 1.7 VI vs. PI in practice: what the two solvers feel like when you run them

Theory: `rl_foundations.md §3.3` (especially the "what iterates" framing and the modified-PI spectrum). This section is a companion — same ideas, but grounded in what actually happens when you call `.solve()` on either solver.

**Same answer, very different feel.** Both solvers terminate with the same $V^*$ and $\pi^*$ for a given MDP. But watching their internals unfold is strikingly different:

- **VI** prints a long sequence of shrinking `delta` values as the Bellman residual crawls toward the tolerance. 100–1000 sweeps is normal for $\gamma$ around 0.9–0.95.
- **PI** prints a tiny number of "outer rounds" (often 3–10), each of which internally ran dozens or hundreds of policy-evaluation sweeps.

The total work often comes out similar — it's been *redistributed* between inner and outer loops.

**A concrete 5×5 grid.** Take the grid world from `rl_foundations.md §3.1` with $\gamma = 0.95$, 24 non-terminal states, and 4 actions. Rough numbers:

- **VI**: ~200 sweeps to hit `delta < 1e-6`. Each sweep is 24 states × 4 actions = 96 Q-evaluations. Total: ~20,000 Q-evaluations.
- **PI**: ~5 outer rounds. Each runs iterative policy evaluation to convergence (~100 sweeps of 24 updates, no `max`) plus one improvement sweep (96 Q-evaluations). Total: 5 × (100 × 24 + 96) ≈ 12,500 updates.

Similar order of magnitude. When $\gamma$ moves closer to 1, VI's sweep count grows sharply (the $\gamma$-contraction gets weaker) while PI's outer round count barely moves — so PI pulls ahead. On problems with large $|A|$ and simple optimal structure, PI can be dramatically faster.

**Where this shows up in the code.** The difference is visible in the `history` tracking:

- `ValueIteration.history` is a list of per-sweep Bellman residuals. Length ≈ number of sweeps.
- `PolicyIteration.eval_iterations` is a list of per-round *evaluation* iteration counts — e.g., `[120, 85, 40, 12, 1]`. That last `1` is the telltale sign of convergence: the policy didn't change on the final round, so evaluation immediately returned.

Read `pi.eval_iterations` from left to right and you can literally see PI finding its way. Early rounds spend a lot of sweeps cleaning up $V^\pi$ for a bad policy; later rounds converge quickly because each new $\pi$ is closer to $\pi^*$ and starts with a $V$ that's already decent from the previous round.

**Which solver does the notebook use for benchmarking?** In `rl_algorithms.ipynb`, the DP reference is VI (not PI). Either gives the same optimal path length, but VI is simpler to reason about for a reference baseline — one tolerance knob ($\theta$) and one cost (sweep count), no inner/outer split.

**The unifying knob (modified PI).** The theory notes (§3.3) explain that PI with $k$ evaluation sweeps per outer round is a spectrum: $k = \infty$ is classical PI, $k = 1$ is essentially VI. The code doesn't currently expose this knob, but to add it you'd just cap `_evaluate_policy` at a small `max_iter` and let the outer loop of `PolicyIteration.solve` do more outer rounds to compensate. The two solvers would converge toward each other as $k \to 1$.

**Takeaway:** VI is "grind on $V$ until convergence, then extract $\pi$." PI is "commit to $\pi$, compute $V^\pi$, improve $\pi$, repeat — using the discreteness of policy space to finish fast." Modified PI is the knob between them. When debugging a DP run, inspecting `history` (for VI) or `eval_iterations` (for PI) tells you a lot about whether the algorithm is making healthy progress or getting stuck.

---

## 2. Deep Q-Networks — `code/agents/dqn.py`

Theory: `rl_in_practice.md §4.3` (why neural Q-learning diverges — the "deadly triad" of function approximation + bootstrapping + off-policy learning) and `§4.4` (DQN's stabilization techniques — target networks and experience replay).

DQN keeps the standard Q-learning update virtually unchanged, but replaces the Q-table with a neural network $Q_\theta(s) \to \mathbb{R}^{|A|}$. Three engineering additions make this stable: input normalization, a replay buffer to decorrelate training samples, and a frozen target network to stop the TD target from drifting as the online network learns. This section covers the code-level nuances of making all those pieces fit together.

### 2.1 Why the state is normalized before going into `QNetwork`

In `_encode`, each `(row, col)` coordinate is divided by the grid's max index to produce inputs in $[0, 1]$:

```python
return torch.FloatTensor([s[0] / self._row_scale, s[1] / self._col_scale])
```

For a tabular agent, none of this would matter — states are dictionary keys, not numeric features. But for a neural network, the state is now a feature vector, and the scale of that vector affects training. Four reasons make the normalization worthwhile, roughly in order of importance:

**1. Weight-initialization schemes assume unit-scale inputs.** PyTorch's `nn.Linear` uses Kaiming/He initialization by default — weights $\sim \mathcal{N}(0, \sqrt{2/n_\text{in}})$ — which is designed so the *variance of activations is preserved* through each ReLU layer. This design only works if inputs come in with roughly unit variance. Feed raw coordinates with row $\in [0, 4]$ and the first-layer pre-activations blow up by a factor of mean² $\approx 4$; with two or three layers, early activations either explode or vanish before a single gradient step. Normalizing to $[0, 1]$ keeps inputs in the range the init scheme was designed for. This is the most subtle reason but arguably the most important.

**2. Gradient scaling across features.** The gradient of a first-layer weight $w_{ij}$ is proportional to the input feature $x_i$ on that connection. If one feature is on scale $[0, 4]$ and another on scale $[0, 0.01]$, the gradients on their respective weights differ by a factor of ~400 per step. Adam's per-parameter adaptive learning rate compensates partially, but you're asking the optimizer to paper over a preprocessing problem. For our grid specifically, both `row` and `col` are on the same scale so this effect is mild here — but it becomes serious the moment you add features on different scales (e.g., position *and* velocity).

**3. Activation-function regime.** For `sigmoid` or `tanh`, large pre-activations saturate the nonlinearity and the gradient vanishes. Our `QNetwork` uses ReLU throughout, which is positively homogeneous (`ReLU(αx) = α · ReLU(x)`) and so doesn't care about scale in this way. But this is why normalization advice is *universal* in deep-learning guides — it matters a lot for non-ReLU architectures.

**4. Grid-size invariance (a practical bonus).** Dividing by `env.rows - 1` means the cell at $(2, 3)$ on a $5 \times 5$ grid gets encoded as $(0.5, 0.75)$, and the same fractional position on a $10 \times 10$ grid also gets $(0.5, 0.75)$. The network sees "position as a fraction of the grid" rather than raw coordinates. This isn't used anywhere in this codebase (we don't transfer weights across grid sizes), but it's a nice property — the same architecture instantiated on any grid size sees inputs in the same distribution.

**Cross-reference with theory.** Your `rl_in_practice.md §4.4` covers target networks and experience replay but doesn't explicitly cover input normalization, because it's a general deep-learning practice, not a DQN-specific stabilizer. For comparison: the original Atari DQN (Mnih et al. 2015) divides raw pixel values by 255 to get inputs in $[0, 1]$ — the same trick, scaled up to $4 \times 84 \times 84$ stacked frames.

**Takeaway:** when a neural network sees a new state, the preprocessing step is not cosmetic — it controls whether the network's first layer operates in the regime its weights were initialized for. For tabular methods this never comes up; once you move to function approximation, it's always a question, and "divide by the max value so inputs live in $[0, 1]$" is the cheapest good answer.

### 2.2 The replay buffer: what it does, why it works, and where it wouldn't

The replay buffer in `dqn.py` is small:

```python
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return s, a, r, s2, done
```

Less than 15 lines, but it's one of the two stabilizers that makes DQN work at all (the other is the target network — see `rl_in_practice.md §4.4`). This section unpacks *why* that is.

**What the buffer is doing mechanically.** The agent is collecting experience online — one transition per environment step. Naively, you'd take that transition and do one gradient step on it immediately. The buffer breaks that coupling: we `push` the transition into a fixed-size FIFO queue and then `sample` a *random minibatch* from the whole queue when it's time to train. The gradient step runs against a uniform sample of recent history, not the current step.

A few implementation notes worth lingering on:

- **`deque(maxlen=capacity)`**: fixed-size FIFO. `append` past capacity drops the oldest element in $O(1)$. Sampling doesn't mutate the deque — `random.sample` picks $k$ distinct random indices and reads them, leaving the queue untouched.
- **We store raw Python tuples `(s, a, r, s2, done)`**, not tensors. If we pushed tensors that were still attached to the autograd graph, every stored transition would carry a computation graph forward indefinitely and memory would balloon. Keeping storage as plain ints/floats/tuples sidesteps this entirely — tensors are constructed only inside `_train_step` when a minibatch is drawn.
- **`zip(*batch)`**: the idiomatic "transpose a list of tuples" trick. A batch of 64 transitions is a list of 64 five-tuples; `zip(*batch)` turns it into five tuples of 64 elements each (one tuple per field). That's exactly the shape `_train_step` wants.

**Why we want a buffer at all — three distinct benefits.**

1. **Sample decorrelation.** SGD's convergence guarantees (and in practice, its stability) assume roughly i.i.d. samples from a stationary distribution. Consecutive environment steps are the opposite of i.i.d. — state $s_{t+1}$ is heavily correlated with $s_t$, rewards come in runs, the agent circles around the same region of state space for dozens of steps. Training on a stream of those correlated samples makes the network over-fit to whatever corner of the space the agent is currently in, then catastrophically forget that region when the agent moves elsewhere. The buffer breaks the correlation by mixing recent and older experience into each minibatch.

2. **Sample efficiency — reusing real experience.** Each real environment step produces exactly one new tuple, but each tuple can be replayed dozens or hundreds of times over its lifetime in the buffer before being evicted. This multiplies the gradient signal per unit of real-world interaction. In expensive simulators (or on real robots) this is the *dominant* reason to use replay. This is the same motivation as Dyna — see below.

3. **Smoothing the training distribution.** The distribution of states the agent visits drifts as the policy improves. Pure online training means the network's loss landscape is also drifting underneath it — a moving target. The buffer averages over a window of recent policies, giving the network a slower-moving, more stable training distribution to fit.

**Why this works for DQN — the off-policy property.** The replay buffer is a tool for *reusing* old experience, but old experience was generated by an *older* policy. For that reuse to be sound, the learning algorithm has to be indifferent to which policy produced the data. Q-learning has this property by construction:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

The target — $r + \gamma \max_{a'} Q(s', a')$ — depends on the *greedy* policy with respect to the current $Q$, not the policy that produced the transition. Q-learning is **off-policy**: it learns about the greedy policy while following something else (ε-greedy, or whatever policy generated the buffer contents). Replay is the dividend you collect from this property.

SARSA's update uses $Q(s', a')$ where $a'$ is the action the agent *actually took next* — i.e., the sample's bootstrapping target is tied to the behavior policy. Reusing old SARSA transitions trains you on targets from an outdated policy, which is a different objective than the current one. So deep SARSA + replay is mathematically unsound in the naive form; it needs importance-sampling corrections to fix, at which point you've given up most of the sample-efficiency win. The on-policy lineage in deep RL mostly skipped this route and jumped to policy-gradient methods instead (REINFORCE → A2C → PPO), which solve the sample-efficiency problem differently.

**Is this like Dyna?** Yes, and the analogy is worth drawing explicitly. Both replay and Dyna are "get more gradient signal per real-environment step," but they do it by different mechanisms:

| | Dyna-Q | Replay buffer |
|---|---|---|
| What's stored | Learned model $\hat T(s' \mid s, a)$, $\hat R(s, a)$ | Raw observed transitions $\langle s, a, r, s', \text{done} \rangle$ |
| Extra updates use | *Simulated* transitions from the model | *Real* past transitions, verbatim |
| Can hallucinate unseen states? | Yes (if the model generalizes) | No — only states you've actually visited |
| Overall approach | Model-based planning | Model-free experience reuse |

A useful reframing: **the replay buffer is non-parametric Dyna.** Instead of fitting a parametric transition model from experience, you keep the transitions themselves *as* your model — your "model" is the empirical distribution over past tuples, and "planning" is sampling from it. This framing appears occasionally in the literature (Sutton has noted the connection).

The tradeoff: Dyna can generalize to states you haven't visited (if its model is smooth); replay cannot. But learning a transition model is *hard* in high-dimensional environments, and bad models produce hallucinated data that's worse than useless. Replay sidesteps the model-learning problem entirely, which is a big part of why deep RL went almost entirely model-free for its first decade.

**The `done` flag is load-bearing.** One subtle but critical detail: we push `done` alongside each transition, and `_train_step` uses it to mask the bootstrap:

```python
targets = rewards + self.gamma * next_q_values * (1 - dones)
```

When `done=True`, the episode ended at $s$ — there is no next state, and the correct TD target is just the terminal reward, not $r + \gamma \max_{a'} Q(s', a')$. Forgetting to mask the bootstrap at terminals is one of the most common DQN bugs. It causes terminal states to inherit a phantom tail of future rewards, which then propagates backwards and corrupts the whole value function.

Because the buffer stores raw transitions, not the policy or any history, we *have to* store `done` explicitly. The buffer has no way to reconstruct it at sample time — by then the episode is long gone.

**Design choices and their alternatives.** Every line of `ReplayBuffer` encodes a design decision. This table captures the main ones and what you'd change to explore a different point in the design space.

| Decision | Our choice | Common alternatives | Why our choice is reasonable |
|---|---|---|---|
| Sampling distribution | Uniform random | Prioritized Experience Replay (PER) — weight by TD error; more recent = higher weight | Uniform is simple and unbiased. PER gives 2–3× sample-efficiency wins but adds a sum-tree data structure, importance-sampling weights, and a priority hyperparameter. Not worth it for this tutorial codebase. |
| Eviction policy | FIFO (drop oldest) | Reservoir sampling (uniform over all history); priority-based eviction | FIFO is the standard. The implicit claim is "old enough experience is likely stale (from a much worse policy) and not worth keeping." Works well when the policy improves steadily. |
| Capacity | 10,000 | 1M in original Atari DQN; smaller (1K–10K) for tiny environments | A gridworld episode is ~5–50 steps, so 10K holds hundreds of episodes — enough for decorrelation, not so much that very stale transitions dominate. Atari needs 1M because episodes are much longer. |
| Minimum buffer size before training | (implicit: batch size) | Often warm up to e.g. 1000 transitions before first gradient step | We start training as soon as there are `batch_size` transitions. For a small problem this is fine; for Atari you'd want a longer warmup so early gradient steps see diverse experience. |
| Storage format | Python tuples in a deque | Preallocated NumPy arrays with a write pointer; circular ring buffer of tensors | Tuples in a deque are the simplest thing that works. NumPy arrays are ~10× faster for huge buffers, but only matter when sampling is your bottleneck. |
| What's stored per transition | `(s, a, r, s', done)` | Add: log-prob of action taken (for off-policy correction); n-step returns; hidden state (for recurrent agents) | The five-tuple is the minimum for one-step Q-learning. Other algorithms need more. |

**Where `replay.push` is called.** Inside `DQNAgent.single_step`:

```python
self.replay.push(s, a, reward, s2, done)
if len(self.replay) >= self.batch_size:
    self._train_step()
```

One real step → one buffer insert → (once warmed up) one minibatch gradient step. Note that "one gradient step per environment step" is another knob: in the original DQN this ratio was 1:4 (one gradient step per four env steps); in some modern setups it's 4:1 (four gradient steps per env step). Tweaking this ratio trades compute for sample efficiency.

**Takeaway.** The replay buffer is a tiny piece of code doing three jobs at once: decorrelating training samples so SGD behaves, multiplying the gradient signal per real step, and smoothing the drifting training distribution as the policy improves. It only works because Q-learning's update is off-policy — the bootstrapping target is defined in terms of the *greedy* policy on the current Q, so stale transitions are still valid training data. The one detail that looks boring but will silently break everything if you forget it is storing `done` and masking the bootstrap at terminals.

### 2.3 Action selection: ε-greedy decay and `torch.no_grad()`

```python
def _select_action(self, s):
    self.epsilon = max(
        self.epsilon_end,
        self.epsilon_start - self.total_steps / self.epsilon_decay
            * (self.epsilon_start - self.epsilon_end))

    if random.random() < self.epsilon:
        return random.randint(0, self.n_actions - 1)

    with torch.no_grad():
        q_vals = self.q_net(self._encode(s))
        return q_vals.argmax().item()
```

Two concerns: decay the ε schedule, then flip a coin to decide explore vs. exploit. Most of this is standard ε-greedy — the two things worth pinning down are the shape of the decay and what `torch.no_grad()` does.

**The decay formula.** Let $t$ = `self.total_steps`, $\varepsilon_0$ = start, $\varepsilon_\infty$ = end, $T$ = decay horizon. The code computes

$$\varepsilon(t) = \max\!\left(\varepsilon_\infty,\ \varepsilon_0 - \frac{t}{T}(\varepsilon_0 - \varepsilon_\infty)\right)$$

— a straight line from $\varepsilon_0$ at $t = 0$ down to $\varepsilon_\infty$ at $t = T$, clamped at $\varepsilon_\infty$ thereafter. The `max(...)` is the clamp; without it the formula would keep going negative.

**Why ε never reaches 0.** Leaving a small residual exploration rate (typically 1–10%) forever serves two purposes:

1. *Continued buffer freshness.* If ε went to 0, the buffer would stop receiving non-greedy transitions, and the agent would have no way to recover from a wrong-but-confident Q-function. The TD update can only nudge Q-values toward better actions it actually *samples*; a pure-greedy agent stuck in a bad basin has no mechanism to escape.
2. *Non-stationarity insurance.* In a fixed environment this is mostly cosmetic, but in anything with drift (dynamics change, rewards are noisy) the residual exploration is what keeps the agent adapting.

**Why decay by *steps*, not episodes.** Early episodes are short (agent wanders near the start) and late-training episodes are either short (converged) or long (stuck). Step-based decay gives ε a predictable trajectory per unit of experience, independent of episode length. Episode-based decay would drop ε fast during early short episodes and slow during mid-training longer ones — almost always the wrong shape.

**`with torch.no_grad()` — what it actually does.** This idiom shows up four times in `dqn.py` and it's worth understanding precisely.

By default, every tensor operation in PyTorch records itself in the **autograd graph** — a computational DAG used later by `.backward()` to compute gradients. For the training forward pass in `_train_step`, the graph is essential. For *every other* forward pass — action selection, target network evaluation, the `Q` property's table extraction — the graph is dead weight:

- **Memory.** Without `no_grad()`, each action-selection call allocates graph nodes that are never used. Over hundreds of thousands of steps this leaks real memory.
- **Speed.** Forward pass is marginally faster without the graph bookkeeping.
- **Correctness.** If a tensor with an attached graph ever leaked into the replay buffer or anywhere it got stored across calls, its entire computation graph would be pinned alive. `no_grad()` here is belt-and-braces on top of the "store raw values, not tensors" rule from §2.2.

Inside `with torch.no_grad():`, graph construction is turned off. Outputs are the same numbers; there's just no tape recording how they were computed. On block exit, normal autograd resumes.

**The split between behavior and target policies.** One detail that's easy to gloss over: this function is the agent's *behavior policy* (what it actually does — ε-greedy), while `_train_step` computes a TD target against the *greedy* policy. Those being different is exactly the off-policy setup from §2.2. The behavior policy injects exploration so the buffer stays diverse; the target policy is what we're actually trying to learn. Q-learning's off-policy update is what lets these two be different without incurring any bias.

**Design choices and their alternatives.** Every knob in ε-greedy encodes a design decision.

| Decision | Our choice | Common alternatives | Why our choice is reasonable |
|---|---|---|---|
| Decay shape | Linear | Exponential $\varepsilon_t = \varepsilon_0 \rho^t$; step-function (drop ε at milestones); cosine | Linear has two intuitive knobs (horizon $T$, floor $\varepsilon_\infty$) and matches the original DQN paper. Exponential requires tuning $\rho$ against training budget, which is fussier. |
| Minimum ε | Strictly positive (0.05 default) | Decay to 0 | Residual exploration keeps the buffer fed with off-policy data indefinitely, so a miscalibrated Q-function can still recover. Decay-to-0 is fragile — any local optimum becomes permanent. |
| Decay clock | Environment steps | Episodes; gradient updates; wall-clock | Step-based decouples ε from episode length, which is erratic early (random wandering) and late (either converged or stuck). Step count is the most stable "amount of experience" metric. |
| Exploration distribution | Uniform (ε-greedy) | Boltzmann ($p(a) \propto \exp(Q(s,a)/\tau)$); noise on parameters (e.g., NoisyNets); Thompson sampling | Uniform is the simplest thing that works. Boltzmann explores toward what Q already likes, which is useful late but actively harmful early when Q is still noise. |
| Initial exploration rate | $\varepsilon_0 = 1.0$ | Start at 0.5 or lower; optimistic Q-value initialization instead of high ε | Starting at 1.0 gives a fully random policy for the first episodes, which is the cleanest way to seed the buffer with diverse transitions before the network has learned anything. |
| Global ε (not state-dependent) | Yes | Per-state counts (UCB-style); RND (Random Network Distillation) for intrinsic rewards | Per-state exploration bonuses matter a lot on hard-exploration problems (Montezuma's Revenge). On gridworld they're overkill. |

**A subtle point about the formula itself.** Reading the code it's tempting to assume `self.epsilon` is computed *per call* deterministically from `total_steps`, which means there's no hidden state: you could delete `self.epsilon` and recompute it on demand and the behavior would be identical. The reason it's stored as an attribute is so that `show()` and verbose logs can display it without recomputing. Minor, but worth knowing — it means there's no "ε-greedy state" other than the step counter.

**Takeaway.** The two substantive things in this function are the linear decay (with clamp) and `torch.no_grad()`. Everything else is plumbing. The clamp ensures exploration never dies; `no_grad()` ensures we're not building a useless autograd graph every time the agent picks an action. The separation between behavior (ε-greedy) and target (greedy) policies is the quiet off-policy assumption that makes the whole DQN setup work.

### 2.4 The training step: `gather`, semi-gradients, and Double DQN

`_train_step` is the heart of DQN — where the replay buffer gets consumed and the network actually learns. It's ~30 lines, but every line is doing something worth understanding.

```python
def _train_step(self):
    if len(self.replay) < self.min_replay:
        return

    states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
    s_t = self._encode_batch(states)
    s2_t = self._encode_batch(next_states)
    a_t = torch.LongTensor(actions)
    r_t = torch.FloatTensor(rewards)
    done_t = torch.FloatTensor(dones)

    q_values = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if self.double:
            best_actions = self.q_net(s2_t).argmax(dim=1)
            next_q = self.target_net(s2_t).gather(
                1, best_actions.unsqueeze(1)).squeeze(1)
        else:
            next_q = self.target_net(s2_t).max(dim=1).values
        td_target = r_t + self.gamma * next_q * (1 - done_t)

    loss = self.loss_fn(q_values, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

Four pieces worth isolating: the warmup guard, the `gather` pattern, the semi-gradient trick (`no_grad` around the target), and Double DQN's selection/evaluation split.

**The warmup guard.** The first thing `_train_step` does is bail out if the buffer has fewer than `min_replay` transitions:

```python
if len(self.replay) < self.min_replay:
    return
```

Early in training, sampling a 32-element minibatch from, say, 40 transitions would give you severely overlapping samples, which defeats the decorrelation purpose of the buffer (§2.2). The guard ensures minibatches are drawn from a population at least `min_replay` in size. A practical consequence: the first `min_replay` environment steps are "pure rollout" — ε is at its starting value, the buffer is filling up, but the network is completely untouched. If you log `train_losses`, you won't see any entries until step `min_replay`.

**The `gather` pattern.** This line is the trickiest bit of tensor manipulation in the file:

```python
q_values = self.q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
```

The network outputs shape `(B, n_actions)` — a full row of Q-values per state. But the Q-learning update only touches *one* Q-value per transition: the one for the action actually taken. We need to pick out `Q[i, actions[i]]` for each row $i$ in the batch.

Step by step:
1. `self.q_net(s_t)` → shape `(B, n_actions)`.
2. `a_t.unsqueeze(1)` → shape `(B, 1)`. `gather` needs its index tensor to match the rank of the input tensor along the gathering axis.
3. `.gather(1, index)` → picks `input[i, index[i, 0]]` for each row. Shape `(B, 1)`.
4. `.squeeze(1)` → drops the trailing `1`, giving `(B,)`.

Net effect: a vector of $B$ Q-values, one per sampled transition, for the action that was actually taken. This is the **prediction** side of the MSE loss.

A NumPy equivalent would be `q_all[np.arange(B), actions]` (advanced indexing). PyTorch provides `gather` as the canonical autograd-compatible way to do this. Note this forward pass is *outside* `no_grad()` — these are exactly the Q-values we'll backprop through.

**The `(1 - done_t)` mask.** Already covered in §2.2 but worth reiterating because this is its natural home:

$$y = r + \gamma \cdot \text{next\_q} \cdot (1 - \text{done})$$

When `done=1`, the bootstrap term vanishes and the target is just $r$. Forgetting this mask lets terminal states inherit a phantom tail of future rewards, which propagates back through the network and corrupts everything. It's a one-character bug with outsized consequences.

**Semi-gradients: why `no_grad()` around the target is theoretical, not just an optimization.** The `with torch.no_grad():` wrapping the target computation is doing real conceptual work here — far more than the `no_grad()` in `_select_action` (§2.3). Understanding this is the difference between "DQN is black magic" and "DQN is standard Q-learning with a network."

Q-learning's update is:

$$\theta \leftarrow \theta - \alpha \nabla_\theta \tfrac{1}{2}\big(y - Q_\theta(s, a)\big)^2 \quad \text{where}\quad y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

The target $y$ is **treated as a constant** when taking the gradient. Even though $y$ clearly depends on network parameters — the target net's $\theta^-$, and in Double DQN also the online net's $\theta$ — we *do not* differentiate through it. The gradient is only with respect to the prediction side, $Q_\theta(s, a)$.

This is called a **semi-gradient** method (Sutton & Barto, ch. 9). It's a compromise with deep theoretical consequences:

- **"True gradient" alternative** (residual gradient, Baird 1995): differentiate through the target. Result: a proper gradient on the Bellman error, with stronger convergence guarantees... and *dramatically* slower empirical performance. The loss landscape is nastier because the network has to decrease both $Q_\theta(s,a)$ and its own bootstrapped target simultaneously, which is ill-conditioned.
- **"Semi-gradient"** (what DQN does): freeze the target, minimize MSE against it as if it were a supervised regression label. No strict convergence guarantees with nonlinear function approximation, but empirically much faster and the standard choice everywhere.

The target network exists precisely to make this semi-gradient trick work well: by keeping target parameters $\theta^-$ frozen for many steps, the "regression label" $y$ stays fixed long enough for the online network to actually fit it before the label moves again. If you used $\theta$ directly (i.e., the same network for prediction and target), the regression label would move on every gradient step, and you'd be chasing your own tail.

So the combination of `torch.no_grad()` *plus* the separate target network is what implements the semi-gradient update. Remove either one — let autograd flow through the target, or use the online net in the target slot — and you've changed the algorithm meaningfully:

| | Target source | `no_grad`? | What you get |
|---|---|---|---|
| Standard DQN | Target net $\theta^-$ | Yes | Semi-gradient, stable |
| Online net in target | Online net $\theta$ | Yes | Moving target, known to diverge (deadly triad) |
| Residual gradient | Target net $\theta^-$ | No | True gradient, empirically slow |
| No target net, no `no_grad` | Online net $\theta$ | No | Even worse — differentiating through the same net twice |

**Double DQN's selection/evaluation decoupling.** The `if self.double` branch swaps a single tensor operation but changes the algorithm in a principled way:

```python
if self.double:
    best_actions = self.q_net(s2_t).argmax(dim=1)           # online picks
    next_q = self.target_net(s2_t).gather(                   # target evaluates
        1, best_actions.unsqueeze(1)).squeeze(1)
else:
    next_q = self.target_net(s2_t).max(dim=1).values         # target does both
```

Vanilla DQN uses the target net for both *picking* the max action and *evaluating* its Q-value. This creates a feedback loop: any overestimation error in the target net's Q-values gets selected *because* it's overestimated, and then that overestimated value is fed into the target as if it were real. Over many updates, the overestimation compounds and systematically biases the Q-function upward.

Double DQN breaks the loop: the online net picks which action looks best, but the target net (an independent noisy estimate) says how good it actually is. Errors between the two networks tend to cancel rather than reinforce. This is the same decoupling trick that Double Q-learning (Hasselt 2010) introduced for tabular methods — DQN just lifted the idea to function approximation by reusing the already-existing target network as the "second Q-estimator."

A summary table of what selects vs. evaluates:

| | Action selection (argmax) | Action evaluation (value) |
|---|---|---|
| Vanilla DQN | Target net | Target net |
| Double DQN | **Online net** | Target net |
| Tabular Double Q-learning | One of two Q-tables (random) | The other Q-table |

The implementation cost is one extra forward pass through the online network. The benefit is substantial on problems where the Q-function has high variance or where many actions have similar values — exactly the cases where picking the max is vulnerable to noise. On simple gridworlds the two look nearly identical in final performance, but Double DQN converges a little more cleanly.

**How the whole step composes.** Putting it all together, one call to `_train_step` does:

1. **Skip if buffer is small** (warmup guard).
2. **Sample** a minibatch and lift it into tensors.
3. **Predict**: forward pass through online net, `gather` the Q-values at the actions taken. Gradient-enabled.
4. **Target**, under `no_grad`:
   - Vanilla: $y = r + \gamma \max_{a'} Q_{\theta^-}(s', a') \cdot (1 - \text{done})$
   - Double: $y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a')) \cdot (1 - \text{done})$
5. **Minimize** MSE between prediction and target with Adam.

Everything that distinguishes DQN from tabular Q-learning is in steps 3–5: the `gather` pattern only exists because we're batching, the target network and `no_grad()` only exist because the Q-function is a neural network that would otherwise chase its own tail, and the `done` mask is the one concession to the fact that we're now processing transitions in batches rather than one at a time in an episode loop.

**Takeaway.** Under the hood, DQN is doing supervised regression every step: fit $Q_\theta(s, a)$ against a fixed label $y$. The label is constructed from a frozen copy of the network's own output, masked at terminals, and (in Double DQN) decoupled into "which action" and "how good" estimates. The `no_grad()` around the target isn't a performance optimization — it's what makes the update a *semi-gradient*, which is the form that empirically works for nonlinear Q-function approximation. Everything else in the function is tensor-shape plumbing.

### 2.5 Target network updates: the cadence knob

This is the counterpart to §2.4 — §2.4 explained *why* the target network exists (the semi-gradient trick needs a stable regression label), and this section covers *how often we refresh it* and what the tradeoff looks like.

The code is two lines inside `single_step`:

```python
if self.total_steps % self.target_update == 0:
    self.target_net.load_state_dict(self.q_net.state_dict())
```

Every `target_update` environment steps (default 100), hard-copy every parameter from the online network into the target network. `load_state_dict` takes a dict of parameter name → tensor and overwrites the target module's parameters in place. Between updates, `target_net` is literally frozen — it doesn't learn, doesn't drift, doesn't update.

**Why it lives in `single_step`, not `_train_step`.** The cadence is keyed on `total_steps` — environment steps — rather than gradient steps. This matters because `_train_step` has an early-return for the warmup period (§2.4), so keying on gradient updates would start the target-refresh clock later. Keying on environment steps gives a uniform "refresh every 100 steps" cadence throughout training, including during warmup. This is a minor choice but either convention is defensible; what matters is that the schedule is predictable.

**The `target_update` hyperparameter — the real knob.** How often to refresh is a genuine tradeoff:

- **Low `target_update` (say 10):** the target net tracks the online net closely, so the TD target reflects recent learning. But the target moves fast, which is the instability we were trying to avoid. In the limit of `target_update=1`, the target is the online net and we're back to the moving-target problem.
- **High `target_update` (say 10,000):** the target is stable for a long time, so the regression problem is well-conditioned. But the target is also *stale* — the online net is learning faster than the target is refreshing, so much of the gradient signal is pulling toward an outdated value estimate. Training slows down.

The sweet spot depends on: how fast the online net is actually changing, how aggressive the learning rate is, how big the buffer is (older data = slower implicit change in what's relevant), and how hard the problem is. Some concrete reference points:

- Original Atari DQN (Mnih et al. 2015): `target_update = 10,000` with a 1M-transition buffer.
- Our 5×5 gridworld: `target_update = 100`. The problem is tiny so the network learns in a few hundred gradient steps; a refresh interval of 10,000 would never fire during a useful training run.

**Hard updates vs. soft (Polyak) updates.** Our implementation uses hard updates — periodic full overwrites. The alternative is soft updates, which move the target network a tiny fraction of the way toward the online network every step:

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \qquad \text{every step}$$

with a small $\tau$ (typically 0.001–0.01). Tradeoff summary:

| | Hard update (what we use) | Soft (Polyak) update |
|---|---|---|
| Frequency | Every $C$ steps | Every step |
| Change | Complete overwrite | Tiny blend ($\tau \approx 0.005$) |
| Target stability | Frozen between copies, then jumps | Smoothly drifting |
| Knob to tune | $C$ (copy interval) | $\tau$ (blend rate) |
| Where it's used | Original DQN, most value-based deep RL | DDPG, TD3, SAC (continuous control) |
| Behavior character | Long quiet periods, sudden jumps in TD targets | Continuously drifting target |

They're roughly equivalent in "how old" the target's information is on average: a hard update every $C$ steps corresponds to a soft update with $\tau \approx 1/C$. The choice comes down to implementation style. Soft updates avoid the small discontinuity in gradients when a hard copy lands, which some practitioners prefer; hard updates are slightly cheaper (one copy every $C$ steps vs. one weighted blend per step). Modern continuous-control algorithms (SAC, TD3) default to soft; DQN and its direct descendants default to hard.

If we wanted to swap our code to soft updates, it would look like:

```python
tau = 0.005
for target_param, online_param in zip(self.target_net.parameters(), self.q_net.parameters()):
    target_param.data.copy_(tau * online_param.data + (1 - tau) * target_param.data)
```

Note `.data.copy_()` — we're bypassing autograd to mutate the parameter tensor in place. The target network never participates in gradient computation (§2.4), so this is safe.

**How it fits into the four-stabilizer picture.** The target network is one of four engineering additions that separate "the Q-table is just a neural network now" (a nice idea) from "and it actually trains without diverging" (DQN):

1. **Replay buffer** (§2.2) → decorrelates samples, gives SGD i.i.d.-like batches.
2. **Target network** (this section) → freezes the TD target between updates so the regression has a stationary label.
3. **Semi-gradient update** (§2.4) → treats the target as a constant at gradient-step time, not backpropping through it.
4. **Input normalization** (§2.1) → keeps the first layer in the regime its weights were initialized for.

Each is a single-digit-line change, and each on its own is insufficient. Remove any one and DQN either diverges outright or learns so slowly it looks like it's diverged. The `load_state_dict` call discussed here is stabilizer #2 — two lines of code implementing a thirty-year-old insight (Lin 1993, revisited by Riedmiller 2005, made to work at scale by Mnih 2013/2015) about how to do temporal-difference learning with nonlinear function approximators without things blowing up.

**Takeaway.** `target_update` is the main knob for trading off stability against staleness. Too low and the target chases the online net (instability); too high and it lags behind useful learning (slow). The 100-step default here matches the scale of the problem; on real-world DQN it'd be 10× to 100× larger. Soft updates (Polyak averaging) are a common alternative that sidesteps the hard/soft distinction by blending continuously, and are essentially equivalent on average. Either way, the goal is the same: give the online network a slowly-changing regression target to fit.

---
