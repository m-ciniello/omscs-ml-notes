# Experiment Budget & Arm Design Analysis

This analysis determines the optimal arm configuration for the Shop Cash Phase 1 experiment, given a budget of ~$500K–$835K and an eligible pool of ~28M active US users. The goal is to maximize the value of this experiment as a **data collection effort for uplift modeling**, not to maximize sample size for its own sake.

> _Pool size note: The ~28M estimate is based on the opportunity-sizing query (30-day active, US, safe risk, not blocklisted/staff). The original brief also excluded users with an existing cash balance — we propose dropping that filter since most users have zero or near-zero balances. [TODO: confirm exact pool size once balance filter decision is finalized.]_

---

## 1. Cost model

The cost of running one treatment arm is driven by how many users actually redeem the incentive. Not every targeted user converts, and not every converter redeems — so the effective cost per user is much less than the face value of the incentive.

**Cost per treatment arm:**

$$
\text{Cost}_{\text{arm}} = N \times r \times \text{incentive amount}
$$

where $N$ is users per arm and $r$ is the **redemption rate** — the fraction of all targeted users who actually redeem:

$$
r = \underbrace{\text{baseline conv} \times (1 + \% \text{conv})}_{\text{conversion rate in treatment}} \times \; cb
$$

| Variable | Definition |
|----------|-----------|
| baseline conv | Organic conversion rate without incentive (~1% at 14d) |
| % conv | Relative conversion lift from the incentive |
| $cb$ | Fraction of converters who redeem the promotion |
| $r$ | Overall redemption rate (% of all targeted users who redeem) |

---

## 2. Cost per user by tier

Using the mini-BFCM comparables from the original brief:

| Tier | % conv (relative lift) | $cb$ | $r$ (redemption rate) | Expected cost per user |
|:----:|:-----:|:---:|:----:|:-----:|
| $5   | 9.9%  | 15.0% | 0.165% | **$0.0082** |
| $10  | 25.0% | 25.0% | 0.313% | **$0.0313** |
| $15  | 38.2% | 32.5% | 0.449% | **$0.0674** |
| $20  | 56.3% | 40.0% | 0.625% | **$0.1250** |

The $20 tier is 15x more expensive per user than $5 — not because it's 4x the face value, but because higher incentives also drive higher redemption rates.

**Total expected cost per user across all 4 treatment tiers** (assuming equal arm sizes):

$$
\$0.0082 + \$0.0313 + \$0.0674 + \$0.1250 = \$0.2319 \;\text{per user}
$$

The control arm has zero cost (no incentive).

---

## 3. Arm configurations vs. budget

For each candidate design, assuming equal allocation across all arms (including control):

### Configuration A: 4 treatment tiers + control (5 arms)

Tiers: $5, $10, $15, $20 + control. Treatment cost/user = $0.2319.

| Budget | N per arm | Total users (5 arms) | % of eligible pool |
|:------:|:---------:|:--------------------:|:------------------:|
| $500K  | 2,156,000 | 10.8M | 39% |
| $650K  | 2,803,000 | 14.0M | 50% |
| $835K  | 3,601,000 | 18.0M | 64% |

### Configuration B: 3 treatment tiers + control (4 arms)

Tiers: $5, $10, $20 + control (drop $15). Treatment cost/user = $0.1645.

| Budget | N per arm | Total users (4 arms) | % of eligible pool |
|:------:|:---------:|:--------------------:|:------------------:|
| $500K  | 3,039,000 | 12.2M | 43% |
| $650K  | 3,951,000 | 15.8M | 56% |
| $835K  | 5,075,000 | 20.3M | 72% |

### Configuration C: 2 treatment tiers + control (3 arms)

Tiers: $5, $20 + control (endpoints only). Treatment cost/user = $0.1333.

| Budget | N per arm | Total users (3 arms) | % of eligible pool |
|:------:|:---------:|:--------------------:|:------------------:|
| $500K  | 3,752,000 | 11.3M | 40% |
| $650K  | 4,877,000 | 14.6M | 52% |
| $835K  | 6,264,000 | 18.8M | 67% |

---

## 4. Statistical power

The primary metric is **14d order conversion** (baseline ~1%). We compute the minimum detectable effect (MDE) for each configuration using the standard two-proportion z-test with Bonferroni correction for multiple treatment-vs-control comparisons (80% power, two-sided).

$$
\text{MDE (abs)} \approx \frac{(z_{\alpha'} + z_\beta) \sqrt{2p(1-p)}}{\sqrt{N}}
$$

where $\alpha' = 0.05 / (\text{number of treatment arms})$ and $p = 0.01$.

| Configuration | Arms | Bonferroni $\alpha$ | Budget $500K | Budget $650K | Budget $835K |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** (5/$10/$15/$20) | 5 | 0.0125 | **3.2%** rel | **2.8%** rel | **2.5%** rel |
| **B** ($5/$10/$20) | 4 | 0.0167 | **2.6%** rel | **2.3%** rel | **2.0%** rel |
| **C** ($5/$20) | 3 | 0.025 | **2.2%** rel | **2.0%** rel | **1.7%** rel |

**All configurations are extremely well-powered.** The expected lifts from the mini-BFCM comparables are 9.9%–56.3% relative — 3x to 20x larger than the worst-case MDE. Statistical power is not a binding constraint for any design.

---

## 5. The tradeoff: tiers vs. sample size

Since power is abundant everywhere, the real question is:

**How many reward tiers does the uplift model need?**

The allocation optimizer (see [objective_design.md](../uplift_modeling/objective_design.md)) solves a multiple-choice knapsack: for each user, it picks the tier that maximizes net uplift. To do this, we need CATE estimates $\hat{\tau}(x, t)$ for **each tier $t$**. More tiers in the experiment means more data points on the dose-response curve, which means:

- Better estimates of the shape of the curve (concave? linear? threshold?)
- More allocation options for the optimizer in Phase 2
- Ability to detect non-monotonic effects (e.g., $15 underperforming $10 for some users)

**Dropping the $15 tier** (Config B) saves ~$145K at the $835K budget level but loses the ability to distinguish "responds to $10 but not $15" from "responds to $10 and to $15 equally." This distinction matters for the knapsack allocator.

**Dropping to 2 tiers** (Config C) gives only the endpoints — useful for a binary "low vs. high" analysis but insufficient for the multi-tier optimization that is the project's goal.

---

## 6. Recommendation

**Configuration A: all 4 treatment tiers + control, at a budget of ~$650K.**

| Parameter | Value |
|-----------|-------|
| Arms | 5 (control + $5 / $10 / $15 / $20) |
| N per arm | ~2,800,000 |
| Total users | ~14M (50% of eligible pool) |
| Estimated cost | ~$650K |
| MDE (14d conversion) | ~2.8% relative (80% power, Bonferroni-corrected) |

**Why this sweet spot:**

1. **Full dose-response data.** Keeping all 4 tiers maximizes the value of this experiment for the uplift model and the Phase 2 knapsack allocator.

2. **More than enough power.** 2.8M users per arm detects a 2.8% relative lift, while the smallest expected lift ($5 tier) is 9.9% — a 3.5x safety margin.

3. **Leaves budget headroom.** Spending $650K instead of $835K preserves ~$185K for Phase 2, where the model-optimized arm will need its own reward budget.

4. **Reasonable pool coverage.** 14M of 28M eligible users (50%) leaves a clean holdout for future experiments.

If the budget must stay closer to $500K, the same configuration at $500K still works (2.15M/arm, MDE = 3.2% relative). If more budget is available, scaling to 3.6M/arm at $835K gives marginal power gains but the primary benefit is a larger training set for the uplift model.

---

## Appendix: cost verification

Cross-checking against the original brief at 3.6M users per arm:

| Tier | r | 3.6M × r × incentive | Original brief |
|:----:|:---:|:----:|:---:|
| $5  | 0.165% | $29,674 | $29,670 |
| $10 | 0.313% | $112,500 | $112,500 |
| $15 | 0.449% | $242,478 | $242,487 |
| $20 | 0.625% | $450,072 | $450,000 |
| **Total** | | **$834,724** | **$834,657** |

Estimates match within rounding error.
