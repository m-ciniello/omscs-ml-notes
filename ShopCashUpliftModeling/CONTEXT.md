# Shop Cash Uplift Modeling — Project Context

## What is this?

This project builds an ML-driven system to optimize **Shop Cash** reward allocation on the Shop App. Shop Cash is a Shopify incentive program that gives users cash rewards (e.g. $5, $10, $15, $20) to drive incremental purchases. The goal: **give the right user the right reward at the right time**, maximizing incremental GMV per dollar of incentive spend under a budget constraint.

## The objective

For each user $u$ and reward tier $t$, the system maximizes **net uplift** — the incremental GMV a reward generates minus its cost:

$$
\text{Net Uplift}(u, t) = \bigl(\text{E}[\text{GMV} \mid u, t] - \text{E}[\text{GMV} \mid u, \text{control}]\bigr) - c(t)
$$

With a fixed budget, optimal allocation is a **multiple-choice knapsack problem**: choose at most one tier per user to maximize total net uplift across the population. The formal ILP formulation and worked examples are in [`objective_design.md`](uplift_modeling/objective_design.md).

## Architecture

The system has two independent components:

1. **Uplift model (ML)** — estimates the conditional average treatment effect $\hat{\tau}(x, t)$ for every user-tier pair. Trained on Phase 1 experimental data.
2. **Allocation optimizer (OR)** — given CATE estimates and a budget, solves the knapsack to produce the reward assignment.

This separation means the model can be retrained independently of business logic (budget size, reward tiers, strategic priorities), and the optimizer can be tuned without retraining the model.

## Phased approach

| Phase | What | Status |
|:-----:|------|:------:|
| **1** | Randomized experiment — control + 4 reward tiers ($5/$10/$15/$20) across ~14M active US users | Designing |
| **2** | Validation experiment — control vs. random rewards vs. model-optimized rewards | Planned |
| **3+** | Continuous optimization via contextual bandits | Future |

## Folder structure

```
ShopCashUpliftModeling/
├── CONTEXT.md                  ← you are here
├── uplift_modeling/             ← uplift model analysis & code (Phase 2+)
│   ├── objective_design.md     ← problem framing, net uplift objective, knapsack formulation
│   └── images/                 ← figures for objective_design.md
├── curriculum/                 ← background reading on uplift modeling techniques
│   ├── curriculum.md           ← learning roadmap and note index
│   ├── causal_inference_uplift_modeling.md   ← causal foundations + CATE estimation methods
│   ├── uplift_evaluation.md    ← evaluation metrics (Qini, AUUC, calibration)
│   └── images/                 ← figures for curriculum notes
├── first_experiment/           ← Phase 1 experiment design
│   ├── experiment_breif.md     ← experiment brief (audience, arms, metrics, eligibility)
│   ├── budget_analysis.md      ← cost model, arm configurations, power calculations
│   └── opportunity-sizing/     ← population & LTV analysis
│       ├── opportunity_sizing.ipynb   ← Jupyter notebook (segments, retention, power calc)
│       └── queries/            ← BigQuery SQL files
└── ...
```

## Key decisions

- **Broad audience**: all active US users, regardless of prior order history. ~80% have not placed an order in the last year, but including repeat buyers lets us measure uplift across segments and train a richer model.
- **4 treatment tiers + control**: provides dose-response data for the uplift model. Budget of ~$650K supports ~2.8M users per arm.
- **Primary metrics**: 14-day order conversion (for statistical power) and 14-day GMV (for uplift modeling).
- **No promotion-recency filter**: experiment will be timed to avoid overlap with other Shop Cash campaigns.

## Team

- **Owner:** Michel Ciniello (ML Engineer)
- **Data Resource:** Rafael Schulman
- **Team:** Buyer Acquisition
