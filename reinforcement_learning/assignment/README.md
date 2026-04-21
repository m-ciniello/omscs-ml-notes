# CS7641 Reinforcement Learning Assignment (Spring 2026)

Implementation and analysis of four RL algorithms — Value Iteration, Policy Iteration, SARSA, and Q-Learning — on two MDPs (Blackjack-v1 and CartPole-v1), plus a Rainbow DQN ablation study on CartPole for extra credit.

## Repository layout

```
assignment/
├── README.md                          # this file
├── rl_report.tex                      # report source (IEEEtran, paste-ready for Overleaf)
├── RL_Report_Spring_2026_v1-2.pdf     # assignment spec
├── RL_Report_Spring_2026_FAQ_v2.pdf   # clarifications
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
│   ├── run.py                         #   run experiments by name-prefix
│   ├── run_all_experiments.sh         #   full campaign in dependency order
│   ├── make_figures.py                #   regenerate figures from results/
│   └── smoke_test.py                  #   single-seed sanity check across all phases
├── results/                           # experiment outputs (per-seed, pickled + JSON; gitignored, regenerate via run_all_experiments.sh)
└── figures/                           # report-ready figures (committed for convenience; regenerated from results/ via make_figures.py)
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
# 1. Set up environment (Python 3.11+ recommended)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Quick sanity check — runs every phase at 1 seed, ~1–2 min
python scripts/smoke_test.py

# 3. Run the full campaign in dependency order (10 seeds per experiment)
#    Full run takes several hours on a laptop CPU; results are cached, so
#    re-invoking the script is a no-op unless you clear results/.
bash scripts/run_all_experiments.sh

# 4. Regenerate all report figures from stored results
#    (Pre-generated figures are already committed under figures/ for
#    convenience; this step overwrites them from freshly-computed results.)
python scripts/make_figures.py
```

### Running individual experiments

```bash
# Run one experiment family by name-prefix
python scripts/run.py --prefix blackjack_vi_
python scripts/run.py --prefix cartpole_qlearning_nbins_sweep

# Just re-print existing results without running
python scripts/run.py --prefix blackjack_vi --no-run

# List every registered experiment
python scripts/run.py --prefix "" --no-run

# Regenerate only specific figures
python scripts/make_figures.py --list
python scripts/make_figures.py --only bj_dp_convergence cp_dp_nbins
```

Results are written to `results/<experiment_name>/` (one subfolder per experiment, containing a `config.json` snapshot and `seed_<i>/` per-seed subfolders). The runner refuses to overwrite existing results, so to rerun an experiment either delete its folder or pass `--overwrite` to `scripts/run.py`.

See `src/configs/` for the full list of registered experiments (one module per phase).
