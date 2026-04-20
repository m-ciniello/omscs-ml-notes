# Hypotheses & Experiment Plan

This document is the bridge between the assignment rubric (`RL_Report_Spring_2026_v1-2.pdf`) / FAQ (`RL_Report_Spring_2026_FAQ_v2.pdf`) and the experiments registered in the `src/configs/` package. Every experiment on disk exists because it tests one of the predictions below.

The hypotheses are written **ahead of the data**: each is grounded in textbook theory or documented prior work, not in numbers we've already seen. Where our eventual results contradict a prediction, that's an interesting finding to discuss in `ANALYSIS.md`, not a reason to quietly retcon the hypothesis.

## Meta-hypothesis

> **H0.** *The structural properties of an MDP — discrete-vs-continuous state, stochastic-vs-deterministic transitions — predict which algorithm family is appropriate and by how much. An analyst who ignores structure and applies every algorithm uniformly will observe exactly this pattern: VI/PI dominate on the problem whose structure admits a compact model; model-free methods dominate on the problem whose structure defeats naive modelling.*

The two environments are chosen precisely to probe this axis:
- **Blackjack** is inherently discrete and stochastic. An analytical MDP is compact (~280 decision states) and exact. This is DP's home turf.
- **CartPole** is continuous and (nearly) deterministic. There is no natural MDP — any tabular representation is a discretization artefact. This is model-free's home turf.

H1–H5 below are the specific, falsifiable predictions that, taken together, either support or refute H0.

## Specific hypotheses

### H1 — VI vs PI on the analytical Blackjack MDP

> *Both algorithms converge to the same V\* and therefore the same greedy policy, giving identical evaluation returns. PI terminates in fewer **outer iterations** than VI (Howard's combinatorial argument — policy improvement on a finite action set takes finitely many steps regardless of θ). Total compute — measured in Bellman backups — is protocol-dependent: our PI runs full policy evaluation to tolerance θ at every outer iteration, which trades fewer outer steps for many more inner sweeps, so the direction of the total-backup comparison is genuinely a measurement rather than a prediction. At γ = 1, neither the Bellman optimality operator (VI) nor the Bellman expectation operator (PI's PE loop) is a strict contraction, so both must be capped by an iteration budget rather than by θ-convergence.*

This gives us four distinct testable claims:
1. Same optimal policy ⇒ matched eval returns across all seeded variants.
2. PI has strictly fewer outer iterations than VI at every γ < 1.
3. Total Bellman backups (VI sweeps vs PI PE-sweeps summed across outer iterations) has no theoretically preordained winner and is expected to depend on γ.
4. γ = 1 is a qualitatively different regime and may exhibit pathological behaviour (e.g. PI's PE loop failing to converge within a reasonable budget).

The γ-sweep (`blackjack_{vi,pi}_gamma_sweep_*`) and θ-sweep (`blackjack_{vi,pi}_theta_sweep_*`) give us HP #1 and HP #2 for DP; eval return is expected to be γ-invariant on Blackjack (rewards are terminal, so the greedy policy is γ-independent for γ ∈ (0, 1]), which is itself a non-trivial claim worth verifying empirically.

The VI ≡ PI policy-equivalence claim is also checked on the estimated CartPole MDP via `cartpole_{vi,pi}_nbins_sweep_*` at a single γ/θ operating point. We do **not** repeat the γ or θ sweeps there: CartPole's MDP is itself an approximation learned from rollouts, and that approximation error would muddle the clean algorithmic comparison the Blackjack sweeps are designed to isolate. The more interesting CartPole-DP variables — sampling policy and grid resolution — are tested in H2 and H4.

### H2 — Sampling-policy dependence of DP on the estimated CartPole MDP

> *The quality of a VI-derived policy on CartPole is bottlenecked not by DP but by the sampling policy used to estimate the MDP. Random rollouts concentrate mass on near-initial states and quick failures; states visited by a competent controller are under-sampled. Using ε-greedy on top of a trained SARSA policy as the sampling policy biases coverage toward the states a good controller actually visits, producing DP-derived policies that score higher on the **real** (non-estimated) dynamics. The effect should be strongest at fine grids where state coverage is the binding constraint, and weakest at coarse grids where random sampling already sees every bin.*

Causal caveat inherited from H4 (discretization): finer grids are where sampling-policy bias should matter most, but they are also where tabular agents' sample budget runs out. If fine-grid tabular agents underperform the FAQ ceiling, it could be an undersampled-MDP artefact (this hypothesis) **or** a "not enough model-free samples" artefact (H4). We separate these by running the DP-on-estimated-MDP pipeline with a fixed, large rollout budget while varying the sampling policy.

Experiments: `cartpole_vi_nbins_sweep_*` (random-sampling baseline at 4 grids), `cartpole_vi_samples_sweep_*` (fixing grid, varying budget), and `cartpole_vi_trained_eps_{3x3x8x12, 5x5x12x16}_{0p1, 0p3, 0p5, 0p7}` (trained-policy sampling at two fine grids × four ε values).

### H3 — SARSA vs Q-Learning on both MDPs

> *Neither Blackjack nor our CartPole discretization contains a "cliff-walking" structure — i.e., a state region where the on-policy (SARSA) and off-policy (Q-Learning) targets give meaningfully different value estimates because exploration near that region is catastrophically expensive. We therefore expect SARSA and Q-Learning to be essentially interchangeable on both environments: similar learning curves, similar final returns, and similar sensitivities to the shared hyperparameters (α, ε-decay). The interesting knob is the exploration schedule itself, not the on/off-policy distinction.*

The cliff-walking framing is what makes SARSA vs Q-Learning an instructive comparison in Sutton & Barto: it exists because the cliff deliberately puts a catastrophic outcome adjacent to the optimal trajectory. Blackjack rewards are terminal and path-independent; CartPole (in our framing) only rewards survival and there are no asymmetric action costs. Absent that structure, the two algorithms are estimating the same Q\* with slightly different updates, so we expect them to land in the same place.

A sufficiently aggressive ε-decay should hurt both agents about equally — which is exactly what the FAQ warns about ("ε decays to near-zero too early ⇒ premature exploitation"). HP sensitivities (α sweep, ε-decay sweep) are expected to track each other between SARSA and Q-Learning on each environment.

Experiments: `blackjack_{sarsa,qlearning}_default`, matching `alpha_sweep` and `eps_decay_sweep` for both, plus the CartPole counterparts (`cartpole_{sarsa,qlearning}_default`, `*_alpha_sweep`, `*_eps_decay_sweep`).

### H4 — Discretization and env-driven HPs on CartPole

> *Policy quality on CartPole is dominated by the angle / angular-velocity bin resolution, not the cart-position resolution. A (1, 1, 6, 6) grid — no spatial bins at all — can balance short-term; a (3, 3, 6, 6) grid nearly solves the task; grids finer than that face diminishing returns because the model-free sample budget is fixed. CartPole is also notably more γ-sensitive than Blackjack because returns accumulate over long horizons (episodes of up to 500 steps), so γ is a first-class HP here even though it is nearly irrelevant on Blackjack.*

Two distinct claims wrapped together because they both stem from "CartPole-the-tabularized-MDP has different structural properties than Blackjack":

1. **Bin allocation matters more than bin count.** The natural prediction from the physics: the pole angle and angular velocity directly determine whether the episode continues, while cart position is bounded away from failure in normal operation. Resolving the state space should prioritise them.
2. **γ matters here but not on Blackjack.** On Blackjack, γ ∈ (0, 1] gives the same optimal policy (H1); on CartPole the horizon is long and rewards accumulate, so the effective discount window meaningfully changes the policy. We expect γ ≈ 0.99 (FAQ recommendation) to be close to optimal, with γ = 0.95 noticeably worse and γ = 1.0 unstable.

Experiments: `cartpole_{sarsa,qlearning}_nbins_sweep_*`, `cartpole_{vi,pi}_nbins_sweep_*`, `cartpole_{sarsa,qlearning}_gamma_sweep`. Together with the α-sweep from H3, this gives us three validated HPs per tabular CartPole algorithm (FAQ minimum is 2).

### H5 — Rainbow-medium ablation on CartPole (extra credit)

> *On CartPole-v1, each of the four Rainbow-medium components (Double DQN, Dueling, PER, N-step) provides a positive marginal improvement over vanilla DQN when the other three are held off. Their effects approximately compose, so the full Rainbow configuration (all four on) should be the best performer or at most indistinguishable from the best single-component variant within 95% CIs. Ranked individually, we expect PER and N-step to help most (they change the training signal: PER reweights which transitions are learned from, N-step shortens the horizon for credit assignment), with Double and Dueling providing smaller stability-oriented gains (they refine the target, not the data).*

This is the Hessel et al. (2018) clean-sweep ablation pattern applied to a four-component subset (we explicitly drop C51 and NoisyNets as out-of-scope upfront). The "marginal improvements compose" claim is the weak form of the Rainbow paper's finding on Atari; whether it holds on the simpler CartPole benchmark is itself a question worth testing.

Experiments: `dqn_ablation_{baseline, double, dueling, per, nstep, rainbow}`, matched HPs (10 seeds each, same replay, same optimizer, same network sizes).

## Rubric-coverage matrix

| Rubric item (from `RL_Report_Spring_2026_v1-2.pdf`) | Covered by |
| --- | --- |
| "How many iterations does VI vs PI take to converge?" | H1 — outer-iteration count plus **total Bellman backups** |
| "Which method converges faster? Why?" | H1 — answer is protocol-dependent; total-backup plot makes this explicit |
| "Did they produce the same optimal policy?" | H1 — matched eval returns + `02_bj_policy_heatmap.png` side-by-side |
| "How does discretization affect CartPole?" | H4 — bin-allocation sweep across (1,1,6,6) → (5,5,12,16) |
| "SARSA vs Q-Learning: sample efficiency, stability, final return?" | H3 on both environments |
| "Exploration strategies and their effect" | H3 — shared ε-decay sweep across SARSA and Q-L |
| "At least 2 validated HPs per model" (FAQ) | H1 (γ, θ for DP on Blackjack); H2 (grid + sampling policy for DP on CartPole); H3/H4 (α, ε-decay, γ, n_bins for model-free); H5 (component toggles + inherited DQN HPs) |
| "5 seeds with mean ± variability" (FAQ) | Every experiment uses 10 seeds (`seeds=(0, …, 9)`), exceeding the FAQ minimum |
| "Staged HP search, not grid search" (FAQ) | 1-D marginal sweeps anchored on a defensible default (FAQ recommendations for CartPole, a single γ × θ operating point for DP). We do **not** do exhaustive grid search; each HP is swept holding others at the default. |
| "Challenges of using function approximation" (rubric §4.4) | H5 — the Rainbow ablation surfaces exactly the pathologies DQN introduces over tabular methods (replay bias, target drift, bootstrapping with a moving function approximator) and shows which targeted fixes address them |
| "Extra credit Rainbow ablation" | H5 |

## Pruning decisions vs. previous registry

Removed as redundant with the hypothesis set above:
- `blackjack_sarsa_nepisodes_sweep_*`, `blackjack_qlearning_nepisodes_sweep_*` — episode count is a training budget, not a hyperparameter in the rubric sense; the default-run learning curves already demonstrate convergence.
- `blackjack_{vi,pi}_grid_g*_*` (full γ × θ grid, 20 cells per algorithm) — collapsed to two 1-D sweeps: γ at θ = 10⁻⁹ reference, θ at γ = 1.0 reference. The grid's only *unique* finding beyond the two marginals was a γ = 1.0 PE-sweep-count anomaly in PI, which survives as a direct comparison between `blackjack_pi_gamma_sweep_0p95` and `blackjack_pi_gamma_sweep_1p0`.

Kept because every other sweep maps directly to a hypothesis above.
