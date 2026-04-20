# TODO

Running list of follow-ups. Items below are surfaced by the final walkthrough
review; keep this file up to date as we find more.

## Before submission

- [x] Rerun full campaign after the infra cleanup (`bash scripts/run_all_experiments.sh`, ~60 min). *Completed Apr 20, 2026. Log: `.logs/run_all_20260419_192100.log`.*
- [x] Reconcile `ANALYSIS.md` numbers against the re-generated `results/`. *Completed. All table entries, sweep counts, and per-grid numbers now match the regenerated `results/` directly.*
- [x] Call out PI's iterative-evaluation design choice in `ANALYSIS.md` (VI-vs-PI section or a methods footnote). *Added as a "Methods note" paragraph inside H1.*
- [x] Prune redundant sweep points. *Done: Blackjack VI/PI γ/θ sweeps now only keep endpoints; CartPole VI sample-budget sweep dropped the 2000-episode interior point. See `configs.py` docstrings for the rationale.*
- [x] Report DQN ablation variance as "fraction of seeds that hit the 500-step cap" alongside mean ± σ. *CartPole-v1 returns are bimodal so mean ± σ is misleading on its own. Added to H6 table and prose.*

## Nice-to-have / post-submission

- [ ] Unit test: compare `Blackjack.transitions(s, a)` against empirical Gym rollout frequencies over ~1e5 trials per (s, a); assert total-variation distance < small threshold. Protects the hand-written analytical MDP from silently drifting away from the `Blackjack-v1` simulator used by SARSA / Q-learning.
