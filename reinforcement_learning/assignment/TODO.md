# TODO

Running list of follow-ups. Keep this file up to date as we find more.

## Before submission

- [ ] Rerun full campaign at 10 seeds with the new split-configs layout (`bash scripts/run_all_experiments.sh`, ~100-120 min expected).
- [ ] Extend `01_bj_dp_convergence.png` (or add a sibling figure) with a **total-Bellman-backups vs γ** panel comparing VI and PI directly. This is the metric that actually answers the rubric's "which method converges faster" question (outer iterations alone trivialise it — PI always wins by Howard's argument). Requires `sweep_deltas` length for VI and sum of PE-sweeps across outer iterations for PI; both are already logged in `result.pkl` per `src/agents/dp.py`. Aggregate across 10 seeds per γ.
- [ ] Regenerate all figures (`python scripts/make_figures.py`) against the fresh `results/`.
- [ ] Rewrite `ANALYSIS.md` headline numbers against the regenerated `results/`. All current numbers are from the pre-split, partially 5-seed run and are now stale. Move the Rainbow N-step duplicate-transition bug note from the old H6 draft into `ANALYSIS.md` as a "things that went wrong and how we found them" aside under the DQN section.
- [ ] Draft the 8-page IEEE report on Overleaf (MDP descriptions, algorithm derivations, discretization strategy, H1–H5 narrative, AI Use Statement, IEEE bibliography).
- [ ] Prepare `REPRO_RL_<gtid>.pdf` companion sheet with git SHA + READ-ONLY Overleaf link + run instructions.

## Nice-to-have / post-submission

- [ ] Align figure filename prefixes with document figure order. After swapping the H1 figures in `rl_report.tex`, `figures/02_bj_policy_heatmap.png` now renders as Figure 1 and `figures/01_bj_dp_convergence.png` renders as Figure 2. The PDF output is correct (filenames don't show up in the rendered report), but for repo tidiness rename the files on disk, update `scripts/make_figures.py`, and update the `\includegraphics` paths.
- [ ] Unit test: compare `Blackjack.transitions(s, a)` against empirical Gym rollout frequencies over ~1e5 trials per (s, a); assert total-variation distance < small threshold. Protects the hand-written analytical MDP from silently drifting away from the `Blackjack-v1` simulator used by SARSA / Q-learning.
- [ ] Consider a `Study` abstraction for multi-axis experiments (would fold the γ-sweep + θ-sweep pair into one declarative object). Out of scope for this submission; useful for future iterations on the codebase.
