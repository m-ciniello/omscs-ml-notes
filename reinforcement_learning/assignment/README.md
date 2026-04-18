# CS7641 Reinforcement Learning Assignment (Spring 2026)

Implementation and analysis of four RL algorithms — Value Iteration, Policy Iteration, SARSA, and Q-Learning — on two MDPs (Blackjack-v1 and CartPole-v1), plus a Rainbow DQN ablation study on CartPole for extra credit.

## Repository layout

```
assignment/
├── RL_Report_Spring_2026_v1-2.pdf    # assignment spec
├── RL_Report_Spring_2026_FAQ_v2.pdf  # clarifications
├── requirements.txt                   # pinned dependencies
├── src/                               # all source code
│   ├── configs.py                     # single source of truth for experiments
│   ├── agents/                        # RL algorithms (VI, PI, SARSA, Q-Learning, DQN, ...)
│   ├── envs/                          # environment wrappers (Blackjack, CartPole, ...)
│   └── experiments/                   # runner, aggregation, hyperparameter search
├── scripts/                           # executable entry points
├── results/                           # experiment outputs (per-seed, pickled + JSON)
├── figures/                           # report-ready figures (generated from results/)
└── README.md
```

## Design principles

- **Config-as-code.** Every experiment is a named entry in `src/configs.py`. Reproducing a result means running the registered experiment by name; no CLI flag soup.
- **Config-snapshotted results.** Every result directory includes a copy of the exact config used, plus the full RNG seeds and git SHA at run time.
- **Seed-aggregated metrics.** Every reported number is averaged over ≥ 5 independent seeds with variability bands (IQR or 95% CI) as required by the assignment FAQ.
- **Figures from disk.** Figure generation is a separate pass over `results/`; it never re-runs experiments. This makes iterating on plot style free.
- **Minimal dependencies.** Gymnasium for environments, NumPy/PyTorch for computation, matplotlib for plots. No `bettermdptools` — all MDPs are built from scratch.

## Reproducing results

```bash
# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Smoke test (verifies the harness works end-to-end)
python -m scripts.smoke_test

# Run a single named experiment
python -m scripts.run_experiment --name <experiment_name>

# Regenerate all figures from stored results
python -m scripts.regenerate_figures
```

See `src/configs.py` for the full list of registered experiments.

## Status

- [x] Phase 0: infrastructure skeleton (config registry, multi-seed runner, aggregation, smoke test)
- [ ] Phase 1: environments (Blackjack MDP, CartPole discretization) + VI/PI
- [ ] Phase 2: tabular model-free (Q-Learning, SARSA)
- [ ] Phase 3: core experiments + hyperparameter search
- [ ] Phase 4: DQN + Rainbow ablation study (extra credit)
- [ ] Phase 5: report + reproducibility sheet
