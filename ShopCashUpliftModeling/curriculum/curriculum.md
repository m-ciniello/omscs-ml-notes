# Shop Cash Uplift Modeling — Curriculum

## Project goal

Build an ML-driven system that optimizes Shop Cash reward allocation: *who* gets a reward, *how much*, and eventually *when* — maximizing incremental GMV per dollar of incentive spend, subject to a budget constraint.

The immediate deliverable is an uplift model trained on Phase 1 experimental data, paired with a constrained optimizer, and validated in a Phase 2 experiment (control vs. random vs. model-optimized).

## Reference document

- [Objective Design](../uplift_modeling/objective_design.md) — problem framing, the net uplift objective, greedy ROIS heuristic vs. LP-optimal allocation, worked examples.

---

## Notes

### Note 1: Causal Inference & Uplift Modeling
**Status:** reviewed
**File:** `causal_inference_uplift_modeling.md`

The foundational note. Covers both the causal framework and the ML methods for estimating heterogeneous treatment effects.

**Part I — Causal inference foundations** *(motivation)*
- Potential outcomes framework (Rubin causal model)
- ATE vs. CATE: average effects vs. per-user heterogeneous effects
- The fundamental problem of causal inference: only one potential outcome is observed per user
- Why randomization identifies causal effects (and the assumptions it requires)
- Connecting the framework to the Shop Cash Phase 1 experiment

**Part II — Uplift modeling (CATE estimation)** *(methods)*
- Why standard supervised learning doesn't directly apply — predicting a *difference* between unobserved quantities
- **Meta-learners**: S-learner, T-learner, X-learner — bias-variance tradeoffs of each
- **Causal forests** (Generalized Random Forests)
- **Multi-treatment extensions**: adapting methods from binary treatment to 4 tiers + control
- Practical modeling: feature engineering, loss functions, handling outcome distributions

---

### Note 2: Uplift Evaluation
**Status:** reviewed
**File:** `uplift_evaluation.md`

How to evaluate a model when you never observe the ground-truth individual treatment effect.

- Why standard ML metrics (RMSE, AUC) break down for uplift
- **Uplift curves and AUUC** (area under the uplift curve)
- **Qini curves and Qini coefficient**
- Calibration of CATE estimates
- Evaluating the *system* (Phase 2 experiment design: control vs. random vs. optimized)

---

### Note 3: Constrained Allocation
**Status:** not started
**File:** `constrained_allocation.md`

The optimization layer: given CATE estimates, assign rewards optimally under a budget.

- Multiple-choice knapsack formulation
- LP relaxation and integrality
- Practical solvers (scipy, cvxpy, OR-Tools)
- Incorporating uncertainty into allocation:
  - Risk-averse allocation (penalize high-variance CATE estimates)
  - Exploration-aware allocation (invest in uncertainty reduction)

---

### Note 4: Contextual Bandits *(future)*
**Status:** not started
**File:** `contextual_bandits.md`

The bridge from batch uplift modeling to the continuous RL vision.

- Contextual bandits as the formalization of "right user, right reward, right time"
- Thompson sampling, UCB — exploration vs. exploitation
- The batch uplift pipeline as the first iteration of a bandit loop
- When to graduate from batch to online
