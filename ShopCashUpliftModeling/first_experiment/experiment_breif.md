# Shop Cash Variable Incentive Experiment — Phase 1

**Team:** Buyer Acquisition
**Owner:** Michel Ciniello
**Data Resource:** Rafael Schulman

---

## Goal

Learn how different user segments respond to variable incentive amounts, and collect the causal data required to build an **uplift model** that will optimize future reward allocation.

This experiment is **Phase 1 of a multi-phase pipeline**:

| Phase | What | Why |
|:-----:|------|-----|
| **1 (this experiment)** | Randomized trial: control + 4 reward tiers | Collect causal training data with heterogeneous treatment effects |
| **2** | Control vs. random rewards vs. model-optimized rewards | Validate that the uplift model + allocator outperforms naive targeting |
| **3+** | Continuous optimization (contextual bandits) | Right user, right reward, right time |

This experiment is designed primarily as a **data collection effort**, not a standalone ship-or-not decision. The outputs feed directly into the CATE estimation and constrained allocation pipeline described in the [project curriculum](../curriculum/curriculum.md) and [objective design](../uplift_modeling/objective_design.md).

---

## Hypothesis

If we give active Shop App users a Shop Cash reward of $X off their next purchase (with 7-day expiry), conversion rates will increase — but the magnitude of the lift will vary significantly across user segments and incentive amounts. By testing 4 reward tiers in a randomized experiment, we can:

1. Estimate the **conditional average treatment effect** (CATE) as a function of user features and reward tier.
2. Identify which user segments are **persuadable** vs. sure things vs. lost causes vs. sleeping dogs.
3. Determine the **dose-response curve** — whether the relationship between incentive amount and incremental GMV is concave, linear, or has diminishing returns.
4. Train an uplift model that powers the Phase 2 optimized allocation.

---

## Audience

**All active US Shop App users** meeting safety/eligibility filters. No order history restriction.

### Eligibility criteria

- Active on Shop App in the last 30 days
- US-based
- Safe or low-risk
- Not on giveaways blocklist
- Not Shopify staff
- No explicit promotion-recency filter — we will time the experiment launch to avoid overlap with other Shop Cash campaigns

> **Note on existing cash balances.** The original brief excluded users with any existing balance. We propose **removing this filter** — preliminary analysis suggests the vast majority of active users have zero or near-zero balances, so this filter excludes very few users while adding implementation complexity. _[TODO: pull distribution of existing cash balances among eligible users to confirm. Expected: median = $0, <X% have balance > $0.]_

### Why broaden beyond first-time buyers?

The [original brief](#) restricted eligibility to users with zero prior orders. This experiment removes that restriction for three reasons:

1. **The 1→2 order transition is at least as valuable.** [Opportunity-sizing analysis](opportunity-sizing/opportunity_sizing.ipynb) shows that ~65% of first-time buyers never place a second order — the largest drop-off in the funnel. Median 365-day LTV after a first order is $0; after a second order it jumps substantially. Incentivizing the 1→2 transition may yield higher ROI than acquiring new buyers.

2. **Uplift models need heterogeneity.** The whole point of CATE estimation is to discover *who* responds differently. Restricting to a single segment (zero orders) artificially compresses the feature space and limits what the model can learn. Including 1-order and 2+ order users gives the model purchase-history features that may be strongly predictive of treatment effect.

3. **The composition handles itself.** ~80% of active eligible users have not placed an order in the last year, so the experiment will be naturally dominated by non-buyer data. Including users with recent order history adds diversity without diluting the signal.

**Estimated eligible pool: ~28M users.**

---

## Arm design

5 arms with equal allocation, based on the [budget analysis](budget_analysis.md):

| Arm | Treatment | Incentive | 7d expiry | Notification |
|:---:|-----------|:---------:|:---------:|:------------:|
| 0 | **Control** | None | — | — |
| 1 | $5 reward  | $5 | Yes | Email + push |
| 2 | $10 reward | $10 | Yes | Email + push |
| 3 | $15 reward | $15 | Yes | Email + push |
| 4 | $20 reward | $20 | Yes | Email + push |

### Budget and sample size

| Parameter | Value |
|-----------|-------|
| Arms | 5 (control + $5 / $10 / $15 / $20) |
| N per arm | ~2,800,000 |
| Total experiment users | ~14M |
| Estimated redemption cost | ~$650K |
| MDE (14d order conversion) | ~2.8% relative lift (Bonferroni-corrected, 80% power) |

The expected lifts from mini-BFCM comparables range from 9.9% to 56.3% relative — well above the MDE. See [budget_analysis.md](budget_analysis.md) for the full cost model and alternative scenarios.

### Why 4 tiers (not fewer)?

The Phase 2 allocator solves a **multiple-choice knapsack**: for each user, it selects the tier maximizing net uplift (CATE minus reward cost) subject to a total budget constraint. Having data at all 4 price points gives the uplift model a richer dose-response curve to learn from. Dropping tiers saves budget but reduces the allocator's option set and our ability to detect non-monotonic effects.

### Assignment

Single-batch assignment: all eligible users at experiment launch are randomized into one of the 5 arms. No split exposure — assignment and exposure happen simultaneously.

---

## Metrics

### Primary

| Metric | Why |
|--------|-----|
| **14-day Shop App order conversion** | Statistical power metric: binary, well-powered, comparable to prior experiments |
| **14-day Shop App GMV** | Uplift modeling outcome: the quantity the CATE model will predict; feeds the knapsack allocator |

### Secondary

| Metric | Why |
|--------|-----|
| 7-day order conversion | Contrasted with 14d to assess demand pull-forward vs. true incremental demand |
| Gross profit per user (14d) | Net business impact after COGS |
| Net revenue per user (14d) | Revenue accounting for incentive cost |
| 14-day return rate | Guard against low-quality orders driven by incentive gaming |
| 2nd-week return rate | Delayed returns that don't show up in the first window |
| Number of active days (14d) | App engagement as a leading indicator of retention |

### Guardrails

| Metric | Concern |
|--------|---------|
| **iCAC** | Incremental cost of acquiring a buyer must stay within the 3-year payback period (per Finance LTV projections) |
| **Total redemption cost** | Must not exceed the approved budget |

### Uplift-specific outputs (post-experiment)

These are not experiment metrics per se, but deliverables produced from the experimental data:

- **CATE matrix**: predicted incremental GMV for each user × tier combination (T-learner, causal forest — see [Note 1](../curriculum/causal_inference_uplift_modeling.md))
- **Qini curves and AUUC**: ranking quality of the uplift model (see [Note 2](../curriculum/uplift_evaluation.md))
- **Calibration plot**: predicted vs. observed uplift by decile
- **Optimal allocation**: knapsack solution mapping users to reward tiers under a Phase 2 budget

---

## Observation window

**14 days** primary observation period, consistent with prior promotion experiments for comparability.

The experiment cohorts will continue to be tracked beyond 14 days to assess:

- **Longer-term retention**: do incentivized users come back after the initial purchase, or is the lift purely pull-forward?
- **LTV trajectory**: 30d, 90d, 180d, 365d GMV by arm — critical for understanding whether the incentive accelerates users into a higher-LTV trajectory (particularly the 1→2 order transition).
- **Phase 2 readiness**: model training will use 14d outcomes, but LTV tracking informs the objective function for future iterations.

---

## What comes next

This experiment produces the training data. The pipeline after Phase 1:

1. **Train uplift models** on Phase 1 data using meta-learners (T-learner, X-learner) and causal forests. Evaluate with Qini curves and calibration on held-out folds.

2. **Build the allocator**: solve the multiple-choice knapsack to assign each user an optimal reward tier under a Phase 2 budget constraint.

3. **Run Phase 2 experiment**: three arms — control, random rewards (same budget, uniform tier assignment), and model-optimized rewards. The key test is **Optimized vs. Random** with equalized budget, which isolates the targeting value of the uplift model.

4. **Iterate**: Phase 2 results feed back into model retraining. Long-term, this evolves into a contextual bandit system that continuously learns the optimal policy.

---

## Prior art

This experiment is similar to the previously tested mini-BFCM promotion but differs in:

| Dimension | Mini-BFCM | This experiment |
|-----------|-----------|-----------------|
| Audience | FTB only (no prior orders) | All active users (no order history filter) |
| Framing | Direct ship/no-ship decision | Data collection for uplift modeling pipeline |
| Tiers | $5 / $10 / $20 | $5 / $10 / $15 / $20 |
| UX | Promotion-style | "Shop Cash off your next purchase" |
| Next step | Evaluate iCAC | Train CATE model → Phase 2 optimization |

Cost estimates use mini-BFCM redemption rates as comparables (see [budget_analysis.md](budget_analysis.md)).

---

## Interaction risks

- Mid-funnel promotions (Browse Abandonment, Cart Sync, Abandoned Cart) are expected to be **paused** during this experiment to avoid interaction effects.
- Users who become eligible for other Shop Cash incentives during the 7-day reward window will be excluded from those campaigns.

---

## Sign-off checklist

| Check | Status |
|-------|:------:|
| Experiment setup reviewed by second Data Scientist | |
| Exposure conditions confirmed | |
| Sample size calculation validated | |
| Metrics agreed upon | |
| Budget approved | |
| Data, Product, and Eng aligned on test plan | |
