# Architecture

This document is the map of the codebase. If you want *what the project is*,
read `README.md`. If you want *how it's organized and why*, read this.

## One-paragraph mental model

Every experiment in this project is a **named entry in a registry**
(`src/configs.py`). The runner (`src/experiments/runner.py`) takes that name,
builds the environment and agent from factories, runs one training/eval pass
per seed, and writes each seed's outputs to `results/<name>/seed_<k>/`
alongside a snapshot of the exact config used. Aggregation, hyperparameter
search, and figure generation are all **downstream consumers of the files
under `results/`** — they never re-run experiments. This separation is the
core architectural commitment: runs produce files, files produce everything
else.

## Directory layout

```
assignment/
├── src/
│   ├── configs.py                  # experiment registry (single source of truth)
│   ├── agents/                     # VI, PI, SARSA, Q-Learning, DQN, random
│   ├── envs/                       # Blackjack (analytical MDP), CartPole (+ estimated MDP)
│   └── experiments/                # multi-seed runner + on-disk result loader
├── scripts/                        # executable entry points (run, make_figures, run_all_experiments)
├── results/                        # experiment outputs (one subtree per experiment)
└── figures/                        # report-ready PNGs, generated from results/
```

## The layers, bottom-up

### 1. Environments (`src/envs/`)

| Module              | What it is                                                                 |
| ------------------- | -------------------------------------------------------------------------- |
| `blackjack.py`      | Hand-rolled analytical MDP for Blackjack-v1 (exact transition + reward) + Gym-backed rollouts for model-free agents. |
| `cartpole.py`       | `DiscretizedCartPole` (binned state for tabular) + `ContinuousCartPole` (raw 4-D for DQN). |
| `cartpole_mdp.py`   | `CartPoleEstimatedMDP`: rolls out a sampling policy, counts transitions, produces a tabular `(T, R)` VI/PI can consume. Evaluation still runs on the real env. |
| `__init__.py`       | `build_env(spec)` factory dispatching on `EnvSpec.name`.                   |

**Why two Blackjack representations?** VI/PI need a model; SARSA/Q-Learning
don't. Blackjack's state space is small enough (≈280 states) that a
closed-form MDP is trivial, which gives VI/PI an exact answer. For CartPole
no such closed form exists, so `cartpole_mdp.py` estimates `T, R` from
rollouts — this is what lets us run VI/PI on CartPole at all.

### 2. Agents (`src/agents/`)

| Module            | Algorithm                                                         |
| ----------------- | ----------------------------------------------------------------- |
| `vi.py`           | Value Iteration on a supplied MDP (also exposes shared DP helpers).|
| `pi.py`           | Policy Iteration on a supplied MDP (imports DP helpers from vi.py).|
| `sarsa.py`        | Tabular SARSA with ε-greedy + linear ε decay.                     |
| `q_learning.py`   | Tabular Q-learning with ε-greedy + linear ε decay.                |
| `tabular.py`      | Shared Q-table + ε-greedy + eval utilities.                       |
| `random_agent.py` | Uniform-random baseline.                                          |
| `dqn/agent.py`    | DQN with optional Double / Dueling / PER / N-step (Rainbow ablation). |
| `dqn/network.py`  | MLP Q-net and dueling Q-net.                                      |
| `dqn/replay.py`   | Uniform + prioritized replay buffers.                             |
| `__init__.py`     | `build_agent(spec)` factory.                                      |

**Agent contract (informal, duck-typed).** Every agent implements

```python
run(env, *, n_episodes, eval_episodes, gamma, seed) -> dict
```

returning a `RunResult` dict with these (optional except `eval_*`) fields:

```
train_returns, train_steps   # empty for DP agents
eval_returns, eval_steps     # always set
history                      # per-algorithm diagnostics (residuals, policy changes, …)
policy, Q                    # final learned policy / Q-table
wall_clock_seconds
```

There is intentionally **no `BaseAgent` ABC**. The contract is short, the
runner and aggregator tolerate missing optional fields, and every agent is
small enough that inheritance was pure ceremony.

### 3. Config registry (`src/configs.py`)

The largest single file in the repo, and deliberately so — it's the
**single source of truth** for every experiment. Anything a reviewer needs
to know about what was run, they can find here.

Structure:

- **Spec dataclasses**: `EnvSpec`, `AgentSpec`, `ExperimentSpec` (all frozen).
- **Registry helpers**: `register(spec)`, `get(name)`, `list_experiments()`.
- **Sweep generator**: `register_sweep(base, path, values, ...)` takes a
  base experiment and a dotted-path override (`"gamma"`,
  `"agent.hyperparams.alpha"`, `"env.kwargs.n_bins"`) and emits one variant
  per value. All variants share a naming convention and a parent directory.
- **`override_at_path`**: the one function that knows how to apply an
  override to a spec — reused by `register_sweep` so the grammar is defined
  exactly once.
- **Registered experiments**, grouped by phase (Blackjack DP → Blackjack
  tabular → CartPole tabular → CartPole DP → DQN ablation).

Naming convention:

```
<env>_<agent>_<variant>              # standalone
<env>_<agent>_<hp>_sweep_<value>     # sweep variant (shared prefix)
```

### 4. Experiment infrastructure (`src/experiments/`)

| Module          | Responsibility                                                                                      |
| --------------- | --------------------------------------------------------------------------------------------------- |
| `runner.py`     | Run experiments across seeds (writes per-seed outputs + config snapshot) and load them back (`load_runs`). Cross-seed statistics are computed at the call site because the exact shape differs per figure. |

Results summaries live inline in `scripts/run.py` (a single ~30-line
helper reading the `summary.json` files for every matching experiment and
printing a mean ± 95% CI table).

**Runner output layout.** For a standalone experiment `foo` with seeds
`[0, 1, 2]`:

```
results/foo/
    config.json        # snapshot of ExperimentSpec.to_dict()
    seed_0/
        result.pkl     # full RunResult dict (source of truth)
        summary.json   # scalar-metrics sidecar (grep-friendly)
    seed_1/ …
    seed_2/ …
```

For a sweep `bar_sweep` over `[v1, v2, …]`, variants nest under a shared
parent:

```
results/bar_sweep/
    v1/config.json, seed_0/…
    v2/…
```

### 5. Scripts (`scripts/`)

| Script                  | What it does                                                                       |
| ----------------------- | ---------------------------------------------------------------------------------- |
| `run.py`                | Run every experiment matching a prefix, then print a per-experiment eval stats table (mean ± 95% CI across seeds). |
| `run_all_experiments.sh`| Runs every registered non-ablation phase in dependency order (Blackjack, then CartPole tabular, then CartPole DP, then DQN). |
| `make_figures.py`       | Regenerates every report figure from `results/`. Never re-runs experiments.       |

## Key design decisions

- **Config-as-code, not CLI-as-code.** Every experiment is a named registry
  entry. Scripts take an experiment name (or a prefix), never a long list of
  hyperparameter flags. Reviewers can read `configs.py` to see every number
  that was reported.
- **Runs produce files; files produce everything else.** Aggregation and
  figure generation are pure functions of `results/`. You can delete `src/`
  and still regenerate every figure in the report from the pickled per-seed
  dumps and JSON summaries.
- **Per-seed snapshots.** Every result directory contains a snapshot of its
  own config plus a git SHA. If a future commit changes a hyperparameter,
  older results are not retroactively invalidated — they carry their
  history with them.
- **Duck-typed agent contract.** No base class. The runner documents the
  shape of the returned `dict`; agents produce it. Less indirection for a
  reader tracing code for the first time.
- **Aggregation stats at the call site.** `runner.py` exposes just
  `load_runs`; every figure does its own mean / CI / IQR because the exact
  shape differs per figure. A generic cross-seed helper never pulled its
  weight.
- **Analytical MDP for Blackjack, empirical MDP for CartPole.** Same VI/PI
  agent implementation runs on both — the difference is entirely in the
  `env` layer. This is what makes it meaningful to compare VI/PI against
  SARSA/Q-learning on CartPole at all.

## Reading order for new contributors

1. `README.md` — what the project is and how to run it.
2. `src/configs.py` top of file + one sweep block — to see the registry pattern.
3. `src/experiments/runner.py` — to see how a spec becomes files on disk.
4. One agent (`src/agents/q_learning.py` is shortest) — the `RunResult` contract in practice.
5. `scripts/make_figures.py` top helpers — to see how `results/` becomes plots.

Everything else is specialization of these five.
