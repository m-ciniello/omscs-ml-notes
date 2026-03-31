# Shop Cash Uplift Modeling — Objective Design

## Context

Shop Cash is a rewards program on the Shop App: users receive cash incentives (e.g. $5, $10, $15, $20) to drive incremental purchases. The goal is to optimize *who* gets rewards, *how much*, and eventually *when* — maximizing the return on every dollar spent.

**Phase 1 (data collection).** A randomized experiment with 5 arms — control, $5, $10, $15, $20 — and a ~$500K budget. Rewards expire in 7 days; users are notified via email and push. The purpose is to collect clean causal data on how each reward tier affects downstream behavior, per user.

**Phase 2 (uplift experiment).** Using Phase 1 data, train uplift models that predict heterogeneous treatment effects. Then run a follow-up experiment: control vs. random rewards vs. model-optimized rewards.

**Long-term vision.** Graduate into a continuous RL-style loop: observe user state → select reward action → observe outcomes → update policy.

---

## The objective: net incremental value

The right per-user objective accounts for both the behavioral lift *and* its cost:

$$
\text{Net Uplift}(u, t) = \underbrace{\bigl(\text{E}[\text{GMV} \mid u, t] - \text{E}[\text{GMV} \mid u, \text{control}]\bigr)}_{\text{incremental GMV (uplift)}} - \underbrace{c(t)}_{\text{cost of reward}}
$$

A useful companion metric is **return on incentive spend (ROIS)** — the incremental GMV generated per dollar of reward:

$$
\text{ROIS}(u, t) = \frac{\text{E}[\text{GMV} \mid u, t] - \text{E}[\text{GMV} \mid u, \text{control}]}{c(t)}
$$

ROIS captures budget *efficiency*: two allocations can have the same net uplift, but the one with higher ROIS frees up budget for other users.

---

## Worked example: from counterfactuals to allocation

The table below shows counterfactual outcomes for six users under three incentive levels ($0, $5, $10). The highlighted ROIS cell marks the best tier for each user — ties are broken by choosing the cheaper option to conserve budget.

![Full counterfactual table for six users showing E[GMV | Incentive], Uplift over control, and ROIS. Highlighted cells mark the best tier per user. A second table shows the budget-constrained allocation for a $20 budget.](images/counterfactual_table_full.png)

Several user archetypes emerge:

- **User 1 — inelastic ("sure thing").** Spends $25 regardless. Any reward is pure waste (ROIS = 0%).
- **User 2 — moderate responder.** Best ROIS at $10 (250%), but lower priority than others given limited budget.
- **User 3 — high-efficiency, diminishing returns.** $5 yields 600% ROIS; $10 drops to 550%. The $5 tier is strictly more efficient.
- **Users 4 & 5 — threshold responders.** No response at $5, but strong response at $10 (500% ROIS). These users have a minimum "activation energy" before any behavioral change occurs.
- **User 6 — extremely responsive.** 1400% ROIS at $5 — the single best dollar-for-dollar investment in the pool.

### Greedy ROIS allocation (budget = $20)

A natural heuristic: rank all user-tier options by ROIS, then greedily allocate from highest down until the budget is exhausted.

| Rank | User | Tier | Uplift | Cost | ROIS | Net Uplift | Cumulative Cost |
|:----:|:----:|:----:|:------:|:----:|:----:|:----------:|:---------------:|
| 1 | 6 | $5 | $70 | $5 | 1400% | $65 | $5 |
| 2 | 3 | $5 | $30 | $5 | 600% | $25 | $10 |
| 3 | 4 | $10 | $50 | $10 | 500% | $40 | **$20** |
| — | 5 | $10 | $50 | $10 | 500% | — | *(excluded — budget exhausted)* |
| — | 2 | $10 | $25 | $10 | 250% | — | *(excluded)* |
| — | 1 | — | — | — | 0% | — | *(inelastic — never rewarded)* |

**Result:** $20 spent, $150 in incremental GMV, **$130 net uplift**.

---

## Why greedy ROIS is a heuristic, not the optimal solution

The greedy approach is intuitive and works well for demonstration, but it is **not** guaranteed to find the optimal allocation. It can fail when locking a user into their highest-ROIS tier leaves budget on the table that a different assignment would use more productively.

**A counterexample.** Two users, budget = $15:

| User | Tier | Uplift | Cost | ROIS | Net Uplift |
|:----:|:----:|:------:|:----:|:----:|:----------:|
| A | $5 | $50 | $5 | 1000% | $45 |
| A | $10 | $90 | $10 | 900% | $80 |
| B | $5 | $45 | $5 | 900% | $40 |

Greedy ROIS picks A→$5 (1000%), then B→$5 (900%). Total cost = $10, net uplift = **$85**, and $5 of budget is wasted — no remaining option can use it.

The optimal solution is A→$10, B→$5. Total cost = $15, net uplift = $80 + $40 = **$120**. The greedy approach left $35 on the table because it committed to the highest-ROIS option without considering the system-level allocation.

The core issue: ROIS ranks *efficiency* but ignores the *volume* of value a given spend level unlocks. When budget capacity exists to absorb a slightly less efficient but much higher net-uplift assignment, the greedy approach misses it.

---

## The correct formulation: multiple-choice knapsack

With a fixed budget, the optimal allocation is a **multiple-choice knapsack problem**: each user is a "group," each treatment tier is an "item" in that group, and we select at most one item per group to maximize total net uplift subject to the budget.

$$
\max_{\{x_{u,t}\}} \sum_{u} \sum_{t} x_{u,t} \cdot \bigl[\text{CATE}(u,t) - c(t)\bigr]
$$

subject to:

$$
\sum_{t} x_{u,t} \leq 1 \quad \forall\, u \qquad \text{(each user gets at most one tier)}
$$

$$
\sum_{u} \sum_{t} x_{u,t} \cdot c(t) \leq B \qquad \text{(budget constraint)}
$$

$$
x_{u,t} \in \{0, 1\}
$$

where $\text{CATE}(u,t) = \text{E}[\text{GMV} \mid u, t] - \text{E}[\text{GMV} \mid u, \text{control}]$ is the conditional average treatment effect.

This is technically an **integer linear program** (ILP) because the decision variables $x_{u,t}$ are binary. However, the LP relaxation (allowing $x_{u,t} \in [0, 1]$) is almost always tight for this problem structure — meaning the continuous solution naturally produces integer assignments. In practice, with millions of users and only 4 treatment tiers, this is fast to solve with standard solvers (`scipy.optimize.linprog`, `cvxpy`, or even Google OR-Tools).

---

## Two-step architecture

This naturally splits into two independent components:

1. **Uplift model (ML).** Estimate $\text{CATE}(u, t)$ for every user-tier pair. This is a causal inference problem — the model predicts heterogeneous treatment effects from the Phase 1 experimental data. It does not need to know about the budget; it just produces the best causal estimates it can.

2. **Allocation optimizer (operations research).** Given the CATE estimates and the budget $B$, solve the multiple-choice knapsack to produce the optimal assignment $\{x_{u,t}\}$.

This separation is valuable: the uplift model can be retrained independently of business logic (budget size, reward tiers, strategic priorities), and the allocator can be swapped or tuned without retraining the model.

---

## A note on the right revenue metric

GMV is revenue to *merchants*, not to Shopify. Shopify's actual revenue is a fraction of GMV (the take rate). If the effective take rate is $r$, then the true profit from an incentive is:

$$
\text{Net Profit Uplift}(u, t) = r \cdot \bigl(\text{E}[\text{GMV} \mid u, t] - \text{E}[\text{GMV} \mid u, \text{control}]\bigr) - c(t)
$$

This doesn't change the modeling — just the **decision threshold** for whether a reward is worth giving at all. A $10 incentive at a 2.5% take rate needs to generate $10 / 0.025 = $400 in incremental GMV just to break even. Depending on how this program is accounted for internally, the objective should reflect the metric the business actually optimizes against.
