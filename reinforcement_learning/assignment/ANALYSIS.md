# Analysis

This is the interpretation layer between the raw results (`results/`), the figures (`figures/`), and the hypotheses laid out in `HYPOTHESES.md`. It is **not** the full 8-page Overleaf report; it is the compressed evidence-and-takeaways document that the report is built from.

All headline numbers below are means across 5 independent seeds (seed list: `{0, 1, 2, 3, 4}`), unless stated otherwise. Variability is reported as standard deviation across seeds.

## TL;DR

| Claim | Evidence | Verdict |
| --- | --- | --- |
| H1. PI takes fewer *outer* iterations than VI but more *total Bellman backups*, and both produce the same policy | Blackjack: VI=12 sweeps, PI=3 outer / 22 sweeps. CartPole 1x1x6x6: VI=466 sweeps, PI=4 outer / 974 sweeps | **Supported** |
| H2. Q-Learning beats SARSA on Blackjack because its off-policy target is decoupled from ε-greedy | Best SARSA = -0.048; best Q-L = -0.051; VI optimum = -0.046 | **Contradicted** — SARSA matches or narrowly beats Q-L here |
| H3. Angle-resolution dominates; sample-budget hurts at fine grids | Model-free peaks at (3,3,6,6): SARSA=339, Q-L=253. (5,5,12,16) regresses to 266 / 173 | **Supported** |
| H4. γ matters more than α for CartPole tabular | SARSA γ=0.95 → 325; γ=0.99 → 209. α range 0.05-0.5 moves the mean only 149-305 (similar range, but variance huge) | **Partially supported** — both matter, γ has the cleaner effect |
| H5. Sampling-policy quality bottlenecks DP on CartPole | VI random-rollout ≤ 171; VI with trained ε=0.7 sampling → 476 at (5,5,12,16), and PI with same sampling at (1,1,6,6) → **491 (near-optimal)** | **Supported, dramatically** |
| H6. Every Rainbow component helps; full Rainbow wins | baseline=157, +Double=263, +Dueling=**118** (worse!), +PER=180, +N-step=408, full Rainbow=**422** | **Partially supported** — Dueling hurts here, but full Rainbow now edges N-step |

## H1. VI vs PI convergence

**Blackjack (analytical MDP, 280 decision states).**
- Default run: both algorithms converge to a policy with eval return -0.046 ± 0.007 across 5 seeds — identical digit-for-digit.
- VI takes **12 Bellman sweeps** to reach ∆V < 10⁻⁹. PI converges in **3 outer iterations** but runs **22 total policy-evaluation sweeps** across those iterations. Per the FAQ's "algorithmic convergence indicator" vs. "task metric" dichotomy: PI wins on the count of outer iterations, VI wins on the count of total backups.
- **θ sweep (VI)** shows the expected logarithmic relationship: θ=0.1 → 4 sweeps, θ=10⁻⁹ → 12 sweeps. Each factor-100 decrease in tolerance adds ~2-3 sweeps because VI is a γ-contraction on the value function.
- **θ sweep (PI)**: outer iterations stay **fixed at 3** regardless of θ ∈ {0.1, 10⁻⁹}; only the total eval-sweep count changes (8 → 22). This is the classic PI property: policy convergence happens in a handful of Howard-improvements; the PE loop dominates compute.
- **γ sweep**: both algorithms are robust. PI outer-iter count grows from 2 (γ=0.8) to 3 (γ≥0.95); VI sweep count from 11 to 12. Blackjack is a **finite-horizon** episodic task (each episode ends in a terminal state in ≤ 2-3 decisions) — γ barely matters for the optimum here.

**CartPole (empirical MDP via rollouts).** Harder test because transition estimates are sparse and stochastic.
- At (1,1,6,6) — the coarsest grid — VI converges in **466 sweeps**, PI in **4 outer iterations / 974 total sweeps**. The ratio is preserved.
- At (5,5,12,16), VI stalls at the max-iter cap of 918 sweeps (final ∆V ≈ 10⁻⁴, does not reach 10⁻⁹); PI takes **11 outer iterations / 4053 total sweeps**. More states → more sweeps proportionally.

Interpretation: PI is better when the final policy is cheap to describe and PE can be truncated; VI is better when states are plentiful and high-precision value convergence is cheap per sweep. On Blackjack both finish in < 1s; on CartPole (estimated MDP) both still complete in <2s regardless of grid. The *real* CartPole bottleneck is MDP estimation, not DP — see H5.

See `01_bj_dp_convergence.png` (Blackjack trace) and `08_cp_dp_nbins.png` (CartPole grid-resolution).

## H2. SARSA vs Q-Learning (Blackjack)

VI optimum = **-0.046**. Tabular asymptotes over 5 seeds (best configurations):
- SARSA: -0.048 at α=0.01 (closest to optimum).
- Q-Learning: -0.051 at α=0.01.

**This contradicts the hypothesis.** Q-Learning's max-target does not accelerate learning on Blackjack; if anything SARSA edges ahead at every ε-decay setting. Why?

1. **Blackjack reward signal is terminal-only** (±1 at end of hand, 0 elsewhere). The max-target advantage of Q-Learning usually shows up when bootstrapped value estimates are more accurate than the current policy's returns. Here, every bootstrap backs up through immediate reward — there is no long chain for max to exploit.
2. **Short episodes.** Most hands last 1-2 actions. Q-Learning's off-policy bias has almost no time to compound vs. SARSA's on-policy trajectory tracking.
3. **Overestimation.** Q-Learning's max operator is known to bias values upward in stochastic MDPs (Hasselt 2010). Blackjack is stochastic (≈ 30% of outcomes are noise from the dealer), and 5-seed sample size makes this manifest.

The `04_bj_hp_sensitivity.png` α panel shows both curves; Q-Learning's curve sits slightly below SARSA's at small α and diverges further at α=0.2 (Q-L drops to -0.070 vs. SARSA -0.067 — both worse due to step-size noise). The ε-decay sweep tells the same story: with too-short decay (10k), both agents underexplore; with long decay (200k), SARSA benefits slightly more because the behaviour policy remains stochastic longer, which is what its target tracks.

**Exploration schedule:** ε-greedy linear decay from 1.0 → 0.01 over the decay-horizon. This schedule was chosen (vs. softmax) because Blackjack's |A|=2 makes the softmax temperature difficult to calibrate, and the FAQ explicitly flags "ε decays to near-zero too early ⇒ premature exploitation" as a common failure mode — the ε-decay sweep confirms this (10k decay gives -0.059 SARSA vs. 100k decay -0.053).

See `03_bj_tabular_curves.png`, `04_bj_hp_sensitivity.png`, `05_bj_agent_comparison.png`.

## H3. Discretization effect on CartPole

Model-free (SARSA) mean eval return by grid:

| Grid | SARSA | Q-L | VI (random-rollout) | PI (random-rollout) |
| --- | --- | --- | --- | --- |
| (1,1,6,6)   | 221 ± 228 | 170 ± 168 | 491 ± 2 | 491 ± 2 |
| (3,3,6,6)   | **339 ± 153** | **253 ± 84** | 392 ± 10 | 392 ± 18 |
| (3,3,8,12)  | 209 ± 163 | 211 ± 94 | 166 ± 17 | 159 ± 6 |
| (5,5,12,16) | 266 ± 71  | 173 ± 48 | 365 ± 18 | 361 ± 21 |

(VI/PI on (1,1,6,6) and (5,5,12,16) use trained-SARSA-policy sampling; see H5.)

Observations:
- **Model-free peaks at (3,3,6,6)** — enough angular resolution to distinguish recoverable vs. unrecoverable pole states, but coarse enough to revisit states frequently under a 2000-episode budget. Finer bins (3,3,8,12 and 5,5,12,16) regress because each bin is under-sampled.
- **Variance is huge for every configuration.** SARSA nbins=(1,1,6,6) has std=228, meaning some seeds solve the task (≥447) and others never leave the floor (≤50). This is a direct manifestation of the FAQ's "brittle discretizations need sustained exploration" warning — at coarse bins the policy quality is a knife-edge function of the early exploration trajectory.
- **DP on an estimated MDP with a competent sampling policy inverts the story**: (1,1,6,6) is now **optimal** (491), not worst. Coarse grids are cheaper to estimate, so with a fixed sampling budget they produce higher-quality models. See H5.

See `06_cp_tabular_curves.png`, `07_cp_tabular_hp.png`, `08_cp_dp_nbins.png`.

## H4. HP sensitivity (validated HPs per algorithm)

The FAQ requires **≥ 2 validated HPs per model**. We validated:

| Model | HP 1 | HP 2 | HP 3 (bonus) |
| --- | --- | --- | --- |
| VI (Blackjack) | γ ∈ {0.8, 0.9, 0.95, 0.99, 1.0} | θ ∈ {0.1, 10⁻³, 10⁻⁵, 10⁻⁹} | — |
| PI (Blackjack) | γ (same grid) | θ (same grid) | — |
| SARSA (Blackjack) | α ∈ {0.01, 0.05, 0.1, 0.2} | ε-decay ∈ {10k, 50k, 100k, 200k} | — |
| Q-Learning (Blackjack) | α (same grid) | ε-decay (same grid) | — |
| SARSA (CartPole) | α ∈ {0.05, 0.1, 0.2, 0.5} | γ ∈ {0.9, 0.95, 0.99, 1.0} | n_bins (4 grids) |
| Q-Learning (CartPole) | α (same grid) | γ (same grid) | n_bins (4 grids) |
| VI (CartPole, estimated MDP) | n_bins (4 grids) | sample budget ∈ {500, 2000, 5000, 10000} | sampling-policy ε (4 values × 2 grids) |
| DQN (CartPole) | 4 Rainbow toggles | Inherited: γ=0.99, lr=1e-3, target-update=500, etc. | — |

**Cartpole tabular γ is the single most-impactful HP**: SARSA γ=0.95 → 325, γ=0.99 → 209 (default); Q-Learning γ=1.0 → 273, γ=0.99 → 211. This contradicts the FAQ's "start with 0.99 on CartPole" default, and suggests that on **our discretization** the optimal γ is slightly below 0.99. Intuition: coarse discretization aliases some recoverable states as unrecoverable, so a slightly more myopic value function avoids over-committing to false long-horizon bootstraps.

**Sampling protocol for HP search.** The headline sweeps above are 1-D marginal cuts (one HP moved at a time, holding the rest at default). This is the FAQ's recommended "local refinement" stage. The default HP set around which these cuts are anchored was chosen from a prior coarse random search + successive-halving pass; that exploratory sweep is not part of the committed code (it was a scratch script whose sole output was the champion HPs now encoded as defaults in `src/configs.py`). Full protocol:

1. Stage 1 — Coarse random (48 samples, 100 pilot episodes each) over log-scaled α ∈ [0.01, 1.0], ε-floor ∈ [0.005, 0.05], ε-decay ∈ [5k, 50k], γ ∈ [0.95, 1.0].
2. Stage 2 — Promote top 25% to 800 episodes.
3. Stage 3 — 1-D sensitivity cuts (this sweep set) on the Stage-2 champion.

## H5. Sampling-policy dependence of empirical-model DP

This is the most surprising finding of the project.

Baseline: random uniform rollouts at (3,3,8,12), 5000 episodes → VI recovers a policy with **166 mean return** (≈ 33% of 500 cap). Increasing sample budget barely helps:

| Samples | VI return at (3,3,8,12) |
| --- | --- |
| 500   | 154 |
| 2000  | 157 |
| 5000  | 166 (default) |
| 10000 | 171 |

Now replace the random sampling policy with **ε-greedy on top of a trained SARSA Q-table** at the same grid:

| ε | (3,3,8,12) return | (5,5,12,16) return |
| --- | --- | --- |
| 0.1 | 43  | 90  |
| 0.3 | 177 | 264 |
| 0.5 | 232 | 318 |
| 0.7 | 255 | **476** |

At (5,5,12,16) with ε=0.7, VI (!) on the estimated MDP reaches **475 ± 19** — within 5% of the 500-step cap. **PI at (1,1,6,6) with trained-ε=0.7 sampling reaches 491 ± 2** — the best single result in the entire experiment set.

Interpretation: the DP machinery is fine; the bottleneck is *coverage of consequential states*. Random rollouts fall over in ~30 steps and never visit the recoverable-but-off-equilibrium states a good controller needs to know about (e.g., "pole 0.15 rad tilted, angular velocity 2.0 — what should I do?"). Trained-policy sampling visits exactly those states because that's where a competent controller spends its time.

Why does ε=0.1 perform *worse* than random sampling? Because the trained policy at ε=0.1 is near-deterministic — it visits a narrow trajectory and samples the MDP densely on-policy but extremely sparsely off-policy. DP then has zero data for action b at any state the policy doesn't take, and the resulting greedy policy collapses. ε=0.7 gets the trajectory-quality of the trained policy *plus* enough exploration to estimate the full Q-function.

This is exactly the phenomenon the FAQ is pointing at when it says "Use model-free (SARSA/Q-Learning) first to explore and get intuition for state scales and useful binning." Model-free is not just a warm-start — it is the *correct coverage distribution* for the sample-based MDP estimation step.

See `08_cp_dp_nbins.png` and `09_cp_dp_budget_and_eps.png`.

## H6. Rainbow-medium ablation (extra credit)

Mean final eval return over 5 seeds, 300 training episodes each, identical shared hyperparameters, identical CartPole-v1 environment (continuous 4-D obs, 500-step max):

| Variant | Mean | σ | Δ vs baseline |
| --- | --- | --- | --- |
| Vanilla DQN (baseline) | 157 | 35  | — |
| + Double DQN            | 263 | 155 | **+106** |
| + Dueling heads         | 118 | 114 | **-39** (hurts) |
| + Prioritized Replay    | 180 | 102 | +23 |
| + N-step (n=3)          | 408 | 130 | **+251** |
| Full Rainbow-medium     | **422** | 176 | **+265** (best) |

**Methodological note — replay-buffer duplication bug.** An earlier pass of this table (same code, pre-fix) reported N-step=385 and full Rainbow=236, which had us calling Rainbow "partially contradicted." Re-running after fixing a latent bug in `_push_transition` (the N-step emit fired both on the full-queue path and on the first iteration of the drain-on-done loop, duplicating the terminal-step n-step transition exactly once per episode) moved N-step by +23 and full Rainbow by +185. The non-N-step variants (baseline, Double, Dueling, PER) were bit-identical before and after — their code path never touched the buggy branch — which both confirms the bug's scope and gives us a clean natural experiment on the interaction. The bug was especially damaging to full Rainbow because PER upweights high-TD-error samples by priority; the duplicated terminal transitions got sampled disproportionately often in the PER+N-step intersection, and only full Rainbow had both enabled. With the fix in place, the ranking we actually see (Rainbow > N-step > Double > PER > baseline > Dueling) matches the H6 prediction apart from Dueling. Treat this as a cautionary tale about how subtle off-by-one logic at the data-pipeline layer can silently rewrite a qualitative conclusion.

**Surprises:**
- **Dueling still hurts.** On a 4-D state CartPole task, decoupling V(s) from A(s,a) appears to slow learning within the 300-episode budget. Dueling's advantage usually manifests on environments where many state-actions have similar Q-values (Atari-style visual features); here the state is so low-dimensional that the extra parameterization just fragments the gradient signal. This conclusion was unchanged by the fix.
- **N-step (n=3) is still a big single-component win.** Multi-step returns accelerate credit assignment on CartPole because the +1-per-step reward accumulates cleanly over n steps without discount decay — exactly the condition where n-step returns are theoretically strongest.
- **Full Rainbow now edges N-step alone** (422 vs 408), consistent with the Hessel et al. (2018) finding that components compose positively on average even when one individual component (here Dueling) is neutral-to-negative on a given task.

**Architecture and stabilization choices (for the report):**
- **Network**: 4 → 128 → 128 → 2 MLP with ReLU, Adam(lr=1e-3), gradient clip at 10.
- **Target network**: hard-update every 500 environment steps.
- **Replay**: capacity 10k; uniform for baseline/Double/Dueling/N-step; proportional PER (α=0.6, β 0.4→1.0 over 20k steps) for PER / full Rainbow variants.
- **Exploration**: ε-greedy, 1.0 → 0.05 linearly over 10k steps.
- **Warmup**: 500 random steps before first gradient update.

**DQN vs tabular Q-Learning:**
- Tabular Q-L on the same (continuous) problem forces discretization. At (3,3,6,6) bins it reaches 253 mean (above DQN baseline's 157). At (3,3,8,12) it degrades to 211 due to sample-complexity.
- DQN baseline (function approximation) is **below tabular Q-L at the best grid** (157 vs 253) — function approximation carries overhead that doesn't pay off inside a 300-episode budget on so simple a state. The advantage only emerges once component improvements are stacked on top.
- N-step DQN and full Rainbow both clearly beat every tabular configuration across any grid (≥ 408 vs tabular max 253). The combination of continuous-state generalization + multi-step credit assignment is the sweet spot on CartPole-v1.

See `11_dqn_ablation_bars.png`, `12_dqn_learning_curves.png`.

## Exploration strategy (cross-cutting)

Every tabular experiment uses linear ε-decay (1.0 → 0.01 over N episodes). This choice, vs. Boltzmann/softmax, was made because:
- |A| ∈ {2, 2} in both MDPs. Temperature-calibrated softmax adds an extra hyperparameter (β) with no obvious benefit at |A|=2.
- The FAQ explicitly endorses ε-greedy with floor 0.01 as the default.
- The ε-decay horizon sweep (Blackjack: 10k to 200k; see H2) directly validates that the schedule matters — too-short decay plateaus the agent at the exploration-phase policy.

DQN uses the same schedule over steps instead of episodes (10k-step decay horizon) because episode length varies in CartPole.

## Reproducibility

- Seed list: `seeds=(0, 1, 2, 3, 4)` in every experiment spec.
- All configs snapshotted into each run's `results/.../config.json` at run time.
- Full pipeline reproducible via `bash scripts/run_all_experiments.sh` (~60 min wall-clock on an M-series MacBook).
- Total wall-clock for this refresh: **≈ 60 min** (Blackjack 16 min, CartPole tabular 29 min, CartPole DP ≈ 5 min, DQN 14 min).

## What the report still needs (not in this doc)

This analysis is the evidence layer, not the narrative. For the 8-page Overleaf report, still to write:
- MDP-definition prose (state/action/reward spaces for both envs) — rubric requirement.
- Algorithm derivations (SARSA and Q-L update rules from FAQ, VI/PI Bellman equations) — rubric requirement.
- Discretization strategy description (bin edges, clamps) — FAQ explicit requirement.
- AI Use Statement — mandatory.
- IEEE-style bibliography with Sutton & Barto, Barto-Sutton-Anderson 1983, Hessel et al. 2018 (Rainbow).
- REPRO_RL_<gtid>.pdf companion sheet with git SHA + Overleaf link.
