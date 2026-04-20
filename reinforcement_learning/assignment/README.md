# CS7641 Reinforcement Learning Assignment (Spring 2026)

Implementation and analysis of four RL algorithms — Value Iteration, Policy Iteration, SARSA, and Q-Learning — on two MDPs (Blackjack-v1 and CartPole-v1), plus a Rainbow DQN ablation study on CartPole for extra credit.

## Repository layout

```
assignment/
├── RL_Report_Spring_2026_v1-2.pdf    # assignment spec
├── RL_Report_Spring_2026_FAQ_v2.pdf  # clarifications
├── requirements.txt                   # pinned dependencies
├── src/                               # all source code
│   ├── configs/                       # experiment registry (one module per phase)
│   │   ├── _registry.py               #   schema + sweep/register machinery
│   │   ├── blackjack_dp.py            #   Phase 1: VI/PI on analytical Blackjack MDP
│   │   ├── blackjack_tabular.py       #   Phase 2: SARSA/Q-L on Blackjack
│   │   ├── cartpole_tabular.py        #   Phase 3: SARSA/Q-L on discretized CartPole
│   │   ├── cartpole_dp.py             #   Phase 3b: VI/PI on estimated CartPole MDP
│   │   └── dqn_ablation.py            #   Phase 4: DQN Rainbow-medium ablation
│   ├── agents/                        # RL algorithms (VI, PI, SARSA, Q-Learning, DQN)
│   ├── envs/                          # environment wrappers (Blackjack, CartPole)
│   └── experiments/                   # multi-seed runner + result loader
├── scripts/                           # executable entry points
├── results/                           # experiment outputs (per-seed, pickled + JSON)
├── figures/                           # report-ready figures (generated from results/)
└── README.md
```

## Design principles

- **Config-as-code.** Every experiment is a named entry in the `src/configs/` package (one topic module per phase). Reproducing a result means running the registered experiment by name; no CLI flag soup.
- **Config-snapshotted results.** Every result directory includes a copy of the exact config used (`config.json`). The runner refuses to overwrite existing `result.pkl` files: rerunning means either `rm -rf results/<experiment>` or passing `--overwrite`, so stale numbers can't silently leak into a report.
- **Seed-aggregated metrics.** Every reported number is averaged over 10 independent seeds (`seeds=(0, …, 9)`) with variability bands (IQR or 95% CI), exceeding the FAQ minimum of 5.
- **Figures from disk.** Figure generation is a separate pass over `results/`; it never re-runs experiments. This makes iterating on plot style free.
- **Minimal dependencies.** Gymnasium for environments, NumPy/PyTorch for computation, matplotlib for plots. No `bettermdptools` — all MDPs are built from scratch.

## Reproducing results

All scripts are invoked from the `assignment/` directory.

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run experiments by name-prefix; a per-experiment results table is printed at the end
python scripts/run.py --prefix blackjack_vi_
python scripts/run.py --prefix cartpole_qlearning_nbins_sweep
python scripts/run.py --prefix blackjack_vi --no-run  # just re-print existing results

# Run the full campaign in dependency order
bash scripts/run_all_experiments.sh

# Regenerate report figures from stored results
python scripts/make_figures.py                  # all figures
python scripts/make_figures.py --list           # list available figure keys
python scripts/make_figures.py --only bj_dp_convergence cp_dp_nbins
```

See `src/configs/` (one module per phase) for the full list of registered experiments; `python scripts/run.py --prefix "" --no-run` prints the full registry.

## Status

- [x] Phase 0: infrastructure (config registry, multi-seed runner, aggregation)
- [x] Phase 1: environments (analytical Blackjack MDP, CartPole discretization) + VI/PI
- [x] Phase 2: tabular model-free (SARSA, Q-Learning)
- [x] Phase 3: core experiments + HP sweeps + DP-on-estimated-CartPole-MDP + figures
- [x] Phase 4: DQN + Rainbow ablation study (extra credit)
- [ ] Rerun full campaign at 10 seeds under the refactored configs package
- [ ] Phase 5: report + reproducibility sheet
