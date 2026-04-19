# Hypotheses & Experiment Plan

This document is the bridge between the assignment rubric (`RL_Report_Spring_2026_v1-2.pdf`) / FAQ (`RL_Report_Spring_2026_FAQ_v2.pdf`) and the experiments registered in `src/configs.py`. Every experiment on disk exists to produce evidence for (or against) one of the hypotheses below.

## Meta-hypothesis

> **H0.** *The structural properties of an MDP — discrete-vs-continuous state, stochastic-vs-deterministic transitions — dictate which algorithm family is appropriate and by how much. An analyst who ignores structure and applies every algorithm uniformly will observe exactly this pattern: VI/PI dominate on the problem whose structure admits a compact model; model-free methods dominate on the problem whose structure defeats naive modelling.*

The two environments are chosen precisely to probe this axis:
- **Blackjack** is inherently discrete and stochastic. An analytical MDP is compact (~280 decision states) and exact. This is DP's home turf.
- **CartPole** is continuous and (nearly) deterministic. There is no natural MDP — any tabular representation is a discretization artefact. This is model-free's home turf.

## Specific hypotheses

### H1 — VI vs PI convergence (both MDPs)

> *PI converges in fewer **outer iterations** than VI but performs more total Bellman backups, because each PI iteration runs an inner policy-evaluation loop to high precision. Both recover the same optimal policy up to ties.*

**Evidence:**
- `blackjack_vi_default`, `blackjack_pi_default` — baseline single-point runs with convergence traces.
- `blackjack_vi_theta_sweep` / `blackjack_pi_theta_sweep` — HP #1: θ controls PE tolerance; PI's outer-iteration count should be insensitive to θ, but its total sweep count should scale with 1/θ.
- `blackjack_vi_gamma_sweep` / `blackjack_pi_gamma_sweep` — HP #2: γ controls convergence rate via the contraction factor; both methods slow as γ → 1.
- Analogous CartPole experiments (`cartpole_vi_nbins_sweep`, `cartpole_pi_nbins_sweep`) demonstrate convergence on the *empirically estimated* MDP.

**Figures:** `01_bj_dp_convergence.png`, `02_bj_policy_heatmap.png`, `08_cp_dp_nbins.png`.

### H2 — SARSA vs Q-Learning on Blackjack (discrete, stochastic)

> *Under the same ε-greedy schedule, Q-Learning reaches the VI-optimal policy slightly faster than SARSA because the off-policy max-target is agnostic to the exploration policy; SARSA's on-policy target is pulled toward the ε-greedy behaviour policy. Both asymptote at the same performance, which matches VI.*

**Evidence:**
- `blackjack_sarsa_default`, `blackjack_qlearning_default` — matched HP, matched ε schedule.
- `blackjack_sarsa_alpha_sweep` / `blackjack_qlearning_alpha_sweep` — HP #1: learning rate sensitivity (FAQ guidance: α ∈ [0.05, 0.5]).
- `blackjack_sarsa_eps_decay_sweep` / `blackjack_qlearning_eps_decay_sweep` — HP #2: exploration horizon. FAQ warns: "ε decays to near-zero too early ⇒ premature exploitation". This sweep reveals exactly that failure mode.
- Comparison against `blackjack_vi_default` tests asymptotic-optimality claim.

**Figures:** `03_bj_tabular_curves.png`, `04_bj_hp_sensitivity.png`, `05_bj_agent_comparison.png`.

### H3 — Discretization effect on CartPole (continuous, deterministic)

> *Policy quality on CartPole is dominated by the angle / angular-velocity bin resolution, not the cart-position resolution. A (1,1,6,6) grid — no spatial bins at all — can balance short-term; a (3,3,6,6) grid nearly solves the task; grids finer than that face diminishing returns because the model-free sample budget is fixed. For DP on the estimated MDP, the sample complexity of the estimate itself bottlenecks fine grids.*

**Evidence:**
- `cartpole_sarsa_nbins_sweep_*`, `cartpole_qlearning_nbins_sweep_*` — four grids × 5 seeds each.
- `cartpole_vi_nbins_sweep_*`, `cartpole_pi_nbins_sweep_*` — DP at the same grids.
- `cartpole_vi_samples_sweep_*` — fixing the grid at (3,3,8,12) and sweeping the MDP-estimation sample budget (HP #2 for DP-on-CartPole: sampling rollout count).

**Figures:** `06_cp_tabular_curves.png`, `07_cp_tabular_hp.png`, `08_cp_dp_nbins.png`, `09_cp_dp_budget_and_eps.png`.

### H4 — HP sensitivity on CartPole tabular (HP validation requirement)

> *Tabular SARSA and Q-Learning on CartPole are more sensitive to γ (the discount factor) and n_bins than to α (learning rate) within reasonable ranges. The FAQ recommends γ = 0.99 for CartPole; we test whether γ = 0.95 or γ = 1.0 materially changes outcomes.*

**Evidence:** `cartpole_*_alpha_sweep`, `cartpole_*_gamma_sweep` for both tabular agents. Combined with the n_bins sweep from H3, this gives three validated HPs per tabular algorithm on CartPole (the FAQ minimum is 2).

**Figures:** `07_cp_tabular_hp.png`.

### H5 — Empirical-model DP on CartPole (sampling-policy dependence)

> *The quality of a VI-derived policy on CartPole is bottlenecked not by DP but by the sampling policy used to estimate the MDP. Random rollouts concentrate mass on near-initial states and quick failures; states visited by a competent controller are under-sampled. Using ε-greedy on top of a trained SARSA policy as the sampling policy biases coverage toward the states a good controller actually visits, producing DP-derived policies that score higher on the **real** (non-estimated) dynamics.*

**Evidence:**
- `cartpole_vi_nbins_sweep_*` — random-sampling baseline at 4 grids.
- `cartpole_vi_trained_eps_{3x3x8x12, 5x5x12x16}_{0p1, 0p3, 0p5, 0p7}` — trained-policy sampling at two fine grids across 4 ε values. Tests (a) does it help at all, (b) how much ε is optimal, (c) is the effect grid-dependent?

**Figures:** `08_cp_dp_nbins.png`, `09_cp_dp_budget_and_eps.png`.

### H6 — Rainbow-medium ablation on CartPole (extra credit)

> *On CartPole-v1, each of the four Rainbow-medium components (Double DQN, Dueling, PER, N-step) provides a positive marginal improvement over vanilla DQN, and the full Rainbow (all four enabled) outperforms every single-component variant. Ranked individually, we expect PER and N-step to help most (they change the loss landscape), Double and Dueling less (they're stability refinements).*

**Evidence:** `dqn_ablation_{baseline, double, dueling, per, nstep, rainbow}` × 5 seeds. All six use the same shared hyperparameters; only the component toggles differ. This is the Hessel et al. (2018) "clean-sweep" ablation pattern applied to a subset.

**Figures:** `11_dqn_ablation_bars.png`, `12_dqn_learning_curves.png`.

## Rubric-coverage matrix

| Rubric item (from `RL_Report_Spring_2026_v1-2.pdf`) | Covered by |
| --- | --- |
| "How many iterations does VI vs PI take to converge?" | H1 — `01_bj_dp_convergence.png` |
| "Which method converges faster? Why?" | H1 |
| "Did they produce the same optimal policy?" | H1 + `02_bj_policy_heatmap.png` (side-by-side) |
| "How does discretization affect CartPole?" | H3 |
| "SARSA vs Q-Learning: sample efficiency, stability, final return?" | H2 (Blackjack), plus CartPole extension |
| "Exploration strategies and their effect" | H2 (ε-decay sweep), H4 |
| "At least 2 validated HPs per model" (FAQ) | H1 (γ, θ for DP), H2/H4 (α, ε-decay, γ, n_bins for model-free), H6 (component toggles + inherited DQN HPs for DQN) |
| "5 seeds with mean ± variability" (FAQ) | Every experiment uses `seeds=(0, 1, 2, 3, 4)` |
| "Staged HP search, not grid search" (FAQ) | Pilot staged search archived in `.logs/staged_search.log`; final refinement = the 1-D marginal sweeps above |
| "Extra credit Rainbow ablation" | H6 |

## Pruning decisions vs. previous registry

Removed as redundant with the hypothesis set above:
- `blackjack_sarsa_nepisodes_sweep_*` and `blackjack_qlearning_nepisodes_sweep_*` — episode count is a training budget, not a hyperparameter in the rubric sense; the default-run learning curves already demonstrate convergence.

Kept because every other sweep maps directly to a hypothesis above.
