# Architecture

This document is the map of the codebase. If you want *what the project is*,
read `README.md`. If you want *how it's organized and why*, read this.

## One-paragraph mental model

Every experiment in this project is a **named entry in a registry** (the
`src/configs/` package). The runner (`src/experiments/runner.py`) takes that
name, builds the environment and agent from factories, runs one training/eval
pass per seed, and writes each seed's outputs to `results/<name>/seed_<k>/`
alongside a snapshot of the exact config used. Aggregation, hyperparameter
search, and figure generation are all **downstream consumers of the files
under `results/`** — they never re-run experiments. This separation is the
core architectural commitment: runs produce files, files produce everything
else.

## Directory layout

```
assignment/
├── src/
│   ├── configs/                    # experiment registry (single source of truth)
│   │   ├── _registry.py            # spec dataclasses + registry + sweep generator
│   │   ├── blackjack_dp.py         # VI/PI on analytical Blackjack MDP
│   │   ├── blackjack_tabular.py    # SARSA / Q-Learning on Blackjack
│   │   ├── cartpole_tabular.py     # SARSA / Q-Learning on discretized CartPole
│   │   ├── cartpole_dp.py          # VI/PI on estimated CartPole MDP
│   │   └── dqn_ablation.py         # Rainbow-medium ablation
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
don't. Blackjack's state space is small enough (~280 decision states) that a
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

### 3. Config registry (`src/configs/`)

The **single source of truth** for every experiment. Anything a reviewer
needs to know about what was run, they can find here. Split into a small
machinery layer and one topic module per phase:

| Module                  | Responsibility                                                                 |
| ----------------------- | ------------------------------------------------------------------------------ |
| `_registry.py`          | Spec dataclasses (`EnvSpec`, `AgentSpec`, `ExperimentSpec`, all frozen), `register`/`get`/`list_experiments`, `register_sweep(base, path, values, …)`, `override_at_path`. |
| `blackjack_dp.py`       | VI/PI specs — baselines, γ-sweep, θ-sweep.                                    |
| `blackjack_tabular.py`  | SARSA / Q-Learning specs — baselines, α-sweep, ε-decay-sweep.                 |
| `cartpole_tabular.py`   | SARSA / Q-Learning on the discretized env — baselines and per-HP sweeps.     |
| `cartpole_dp.py`        | VI/PI on the estimated CartPole MDP — nbins-sweep, samples-sweep, trained-ε sampling. |
| `dqn_ablation.py`       | Rainbow-medium ablation (6 component-toggle variants).                        |
| `__init__.py`           | Imports every topic module so registration side-effects fire at package import, then re-exports the public surface. |

**Why split?** The registry grew to the point where a single file was hard to
scan. Each topic module is now short enough to hold in your head, and an
experiment's definition lives next to other experiments of the same kind,
not in a 1,000-line catch-all.

**Single global registry.** `register()` populates a module-level
`EXPERIMENTS` dict that every consumer (runner, figures, run.py) reads from.
A missing import in `__init__.py` means silently-missing experiments, so the
import block at the bottom of `__init__.py` is the one place where
additions need a line.

Naming convention for registered experiments:

```
<env>_<agent>_<variant>              # single-point
<env>_<agent>_<hp>_sweep_<value>     # sweep variant (shared prefix)
```

Sweep variants nest under a shared `results_path_parts` parent so every
seed of every sweep value lives under the same directory tree (see the
runner output layout below).

### 4. Experiment infrastructure (`src/experiments/`)

| Module          | Responsibility                                                                                      |
| --------------- | --------------------------------------------------------------------------------------------------- |
| `runner.py`     | Run experiments across seeds (writes per-seed outputs + a per-experiment config snapshot) and load them back (`load_runs`). Cross-seed statistics are computed at the call site because the exact shape differs per figure. |

**Failure semantics.** `run_spec` refuses to overwrite existing per-seed
`result.pkl` files. If any seed's pickle is already on disk, the runner
raises `FileExistsError` *before* touching the experiment directory or
writing a new `config.json`, so the on-disk snapshot can never drift out of
sync with the per-seed pickles. To rerun: either `rm -rf results/<name>/`
or pass `--overwrite`. This is a deliberate tradeoff — stale numbers can't
silently leak into a report, at the cost of requiring an explicit `rm` for
iteration.

**Runner output layout.** For a standalone experiment `foo` with seeds
`[0, 1, 2]`:

```
results/foo/
    config.json        # snapshot of ExperimentSpec (asdict) — one per experiment, not per seed
    seed_0/
        result.pkl     # full RunResult dict (source of truth)
        summary.json   # scalar-metrics sidecar (grep-friendly)
    seed_1/ …
    seed_2/ …
```

For a sweep `bar_sweep` over values `[v1, v2, …]`, variants nest under a
shared parent:

```
results/bar_sweep/
    v1/config.json, seed_0/…
    v2/…
```

### 5. Scripts (`scripts/`)

| Script                   | What it does                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `run.py`                 | Run every experiment matching `--prefix`, then print a per-experiment eval stats table (mean / std / 95% CI / wall time across seeds). `--no-run` summarises existing results only; `--overwrite` skips the refuse-to-overwrite check. |
| `run_all_experiments.sh` | Six phases in dependency order: Blackjack (DP + tabular), CartPole SARSA, CartPole Q-Learning, CartPole VI, CartPole PI, DQN ablation. The only real dependency is `cartpole_sarsa_*` before `cartpole_vi_trained_eps_*` (trained-ε sampling loads SARSA Q-tables). Phases are independent shell calls — if one halts (`FileExistsError`, crash, etc.), later phases still run; inspect `$MASTER_LOG` per phase. |
| `make_figures.py`        | Regenerates every report figure from `results/`. Never re-runs experiments.       |

Results-summary printing lives inline in `scripts/run.py` (two small
helpers, `_collect_stats` and `_print_results_summary`) rather than in a
library module — the shape is simple enough that an import-level abstraction
would be overkill, and putting it in the script keeps the prefix-scan
behaviour visible at the entry point.

## Key design decisions

- **Config-as-code, not CLI-as-code.** Every experiment is a named registry
  entry. Scripts take an experiment name (or a prefix), never a long list of
  hyperparameter flags. Reviewers can read `src/configs/*.py` to see every
  number that was reported.
- **Runs produce files; files produce everything else.** Aggregation and
  figure generation are pure functions of `results/`. You can delete `src/`
  and still regenerate every figure in the report from the pickled per-seed
  dumps and JSON summaries.
- **Refuse-to-overwrite result snapshots.** The runner raises
  `FileExistsError` if `result.pkl` exists for any seed. Reruns require an
  explicit `rm -rf` (or `--overwrite`). This is the single rule that
  protects the report's numbers from silently going stale after a code or
  config edit — if you forget to clear, you get a loud error, not a
  wrong-but-plausible figure.
- **Per-experiment config snapshot.** `config.json` is written once per
  experiment directory, not per seed (all seeds in a run share the same
  spec by construction). The snapshot is populated *only after* the
  pre-flight existence check passes, so the snapshot can never disagree
  with the per-seed pickles sitting next to it.
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
  SARSA/Q-Learning on CartPole at all.

## Reading order

1. `README.md` — what the project is and how to run it.
2. `src/configs/_registry.py` + one topic module (e.g. `blackjack_dp.py`) — to see the registry pattern and one concrete sweep.
3. `src/experiments/runner.py` — to see how a spec becomes files on disk, including the refuse-to-overwrite check.
4. One agent (`src/agents/q_learning.py` is shortest) — the `RunResult` contract in practice.
5. `scripts/make_figures.py` top helpers — to see how `results/` becomes plots.

Everything else is specialization of these five.
