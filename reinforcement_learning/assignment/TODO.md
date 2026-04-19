# TODO

Running list of follow-ups. Items below are surfaced by the final walkthrough
review; keep this file up to date as we find more.

## Before submission

- [ ] Rerun full campaign after the infra cleanup (`bash scripts/run_all_experiments.sh`, ~60 min).
- [ ] Reconcile `ANALYSIS.md` numbers against the re-generated `results/`. Known staleness points to check:
  - DQN ablations: N-step and Full Rainbow eval means in the doc disagree with what's on disk (Rainbow now beats N-step, doc says otherwise).
  - Blackjack SARSA / Q-learning default rows: doc appears to report optimal-alpha sweep results instead of default-alpha results.
  - CartPole H3 table footnote: claims VI/PI at `(1,1,6,6)` and `(5,5,12,16)` used trained-SARSA-policy sampling, but those runs use random rollouts per `configs.py`.
  - CartPole PI at `(1,1,6,6)` with trained-ε=0.7 sampling: doc claims 491 ± 2; no such experiment is registered. The 491 number comes from PI with random sampling at that grid size.

## Nice-to-have / post-submission

- [ ] Unit test: compare `Blackjack.transitions(s, a)` against empirical Gym rollout frequencies over ~1e5 trials per (s, a); assert total-variation distance < small threshold. Protects the hand-written analytical MDP from silently drifting away from the `Blackjack-v1` simulator used by SARSA / Q-learning.
