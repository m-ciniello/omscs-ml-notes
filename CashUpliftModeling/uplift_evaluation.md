# Uplift Evaluation

Evaluating an uplift model is fundamentally harder than evaluating a standard prediction model. In classification, you predict $\hat{y}$ and check it against the observed $y$. In uplift modeling, you predict $\hat{\tau}(x) = \text{E}[Y(1) - Y(0) \mid X = x]$, but the true individual treatment effect $\tau_i = Y_i(1) - Y_i(0)$ is **never observed** — you only see one of the two potential outcomes. There is no "ground truth" to score against.

This note develops the evaluation tools that work despite this constraint. The key insight: while we can't evaluate *point accuracy* of individual CATE estimates, we can evaluate whether the model **ranks** users correctly — whether users predicted to have high uplift actually generate more incremental value than users predicted to have low uplift. This ranking property is what the allocation optimizer needs, and it's what uplift curves measure.

The note covers three levels of evaluation: **ranking metrics** (uplift curves, Qini curves, AUUC) that assess how well the model orders users by treatment effect, **calibration** that assesses whether the CATE estimates are properly scaled, and **system-level evaluation** through the Phase 2 experiment design.

---

## 1. Why standard ML metrics break down

Before introducing uplift-specific metrics, it's worth understanding precisely *why* familiar metrics fail.

**RMSE / MAE.** The natural idea is to compute $\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (\hat{\tau}_i - \tau_i)^2}$. But $\tau_i = Y_i(1) - Y_i(0)$ is unobserved — we only have $Y_i^{\text{obs}}$, which equals $Y_i(1)$ or $Y_i(0)$ depending on the user's treatment assignment, never both. There is literally no label to compute the error against.

**AUC / precision / recall.** These assume a binary outcome that the model predicts. Uplift is a *continuous* quantity (the treatment effect), and more importantly, it's about the *difference* between two potential outcomes, not the outcome itself. A user with high $P(\text{purchase} \mid \text{treated})$ could have zero uplift if they'd also purchase without treatment (a sure thing). AUC of the treated-group response model tells you nothing about incremental value.

**The transformed outcome trick.** There is a clever workaround (Athey & Imbens 2015): define the **transformed outcome**

$$
Y_i^* = Y_i \cdot \frac{W_i - p}{p(1-p)}
$$

where $W_i \in \{0,1\}$ is the treatment indicator and $p = P(W = 1)$ is the treatment probability. When $p = 0.5$ (balanced experiment), this simplifies to $Y_i^* = 2Y_iW_i - 2Y_i(1-W_i)$, which equals $+2Y_i$ for treated users and $-2Y_i$ for control users. The key property:

$$
\text{E}[Y_i^* \mid X_i] = \text{E}[Y(1) - Y(0) \mid X_i] = \tau(X_i)
$$

So $Y^*$ is an **unbiased estimate of the ITE** — noisy, but unbiased. To see why, expand the conditional expectation (using unconfoundedness and $P(W=1 \mid X) = p$):

$$
\text{E}[Y^* \mid X] = \frac{1}{p(1-p)}\text{E}[Y(W-p) \mid X] = \frac{1}{p(1-p)}\bigl(\text{E}[YW \mid X] - p\,\text{E}[Y \mid X]\bigr)
$$

Now $\text{E}[YW \mid X] = \mu_1(X) \cdot p$ (the outcome under treatment, times the probability of being treated), and $\text{E}[Y \mid X] = p\,\mu_1(X) + (1-p)\,\mu_0(X)$. Substituting:

$$
= \frac{p\,\mu_1 - p[p\,\mu_1 + (1-p)\,\mu_0]}{p(1-p)} = \frac{p(1-p)\,\mu_1 - p(1-p)\,\mu_0}{p(1-p)} = \mu_1(X) - \mu_0(X) = \tau(X)
$$

This means you *can* compute RMSE of $\hat{\tau}$ against $Y^*$, and in expectation it targets the right thing. The problem is variance: $Y^*$ is extremely noisy for individual users (it's based on a single observation), so the RMSE is dominated by irreducible noise and is hard to interpret in absolute terms. The transformed outcome is more useful as a **training target** (the pylift package uses it) than as an evaluation metric.

**Bottom line.** Point-accuracy metrics require a ground truth we don't have. What we *can* evaluate is whether the model's **ranking** of users by predicted uplift is correct — do users scored as high-uplift actually generate more incremental value? This is what uplift curves measure.

The next two sections develop the two standard ranking metrics — the cumulative gain chart and the Qini curve — that answer this question.

---

## 2. Uplift curves and cumulative gain

### The core idea

Suppose you've trained an uplift model and produced CATE estimates $\hat{\tau}(x_i)$ for all users in a held-out test set (which must also be a randomized experiment — treatment and control). Now:

1. **Sort** all users from highest to lowest predicted uplift $\hat{\tau}$.
2. **Walk** down the sorted list, at each point $\phi$ (the fraction of the population targeted so far) computing the **observed uplift** within the top-$\phi$ users.
3. **Plot** the observed uplift as a function of $\phi$.

If the model is good, the top-ranked users should have the highest *actual* treatment effect, so the curve should rise steeply at first (targeting the persuadables) and flatten as you exhaust the high-uplift users and start targeting sure things and lost causes.

### The cumulative gain chart

The **cumulative gain chart** (Gutierrez & Gérardy 2017) formalizes this. Let $\phi \in [0, 1]$ be the fraction of the population targeted (sorted by predicted uplift, highest first). Define:

- $n_t(\phi)$: number of treated users in the top-$\phi$ fraction
- $n_c(\phi)$: number of control users in the top-$\phi$ fraction
- $n_{t,1}(\phi)$: sum of outcomes among treated users in the top-$\phi$ fraction (for binary outcomes this is a count of conversions; for continuous outcomes like GMV it is the total GMV)
- $n_{c,1}(\phi)$: same for control users

The cumulative gain at targeting fraction $\phi$ is:

$$
\text{CumulativeGain}(\phi) = \left(\frac{n_{t,1}(\phi)}{n_t(\phi)} - \frac{n_{c,1}(\phi)}{n_c(\phi)}\right) \cdot (n_t(\phi) + n_c(\phi))
$$

**Reading this formula.** The first factor $\frac{n_{t,1}(\phi)}{n_t(\phi)} - \frac{n_{c,1}(\phi)}{n_c(\phi)}$ is the **observed uplift** (conversion rate difference, or mean GMV difference) among the top-$\phi$ users. The second factor $(n_t(\phi) + n_c(\phi))$ scales it by the number of users targeted, converting from a *rate* to a *count* — the total incremental conversions (or incremental GMV) generated by targeting the top-$\phi$ users.

**What a good model looks like.** At $\phi = 0$, cumulative gain is 0 (you've targeted no one). At $\phi = 1$, cumulative gain equals the total uplift across the entire population (you've targeted everyone, which is the same as the ATE × $N$). A good model front-loads the gain: most of the incremental value is captured in the first 10-20% of users, and the curve flattens (or even dips, if sleeping dogs exist at the bottom of the ranking).

**The random baseline.** A model with no ranking ability (equivalent to targeting users at random) produces a straight line from $(0, 0)$ to $(1, \text{total uplift})$. The value of the uplift model is the area *between* its curve and this random baseline.

[FIG:ORIGINAL — cumulative gain chart (uplift curve) showing three curves: the model's curve (rising steeply then flattening), the random baseline (straight line), and the perfect model (which rises as steeply as theoretically possible). The x-axis is "fraction of population targeted" and the y-axis is "cumulative incremental conversions/GMV". The shaded area between the model curve and the random line represents the model's value.]

The cumulative gain chart is intuitive but has a practical issue: the local denominators $n_t(\phi)$ and $n_c(\phi)$ can be small or unbalanced in the top-ranked segments, making the uplift rate estimate noisy. The Qini curve addresses this with a different normalization.

---

## 3. The Qini curve

The **Qini curve** (Radcliffe 2007) is a closely related ranking metric that normalizes differently to handle unequal or locally imbalanced treatment/control groups.

### Definition

$$
\text{Qini}(\phi) = \frac{n_{t,1}(\phi)}{N_t} - \frac{n_{c,1}(\phi)}{N_c}
$$

where $N_t$ and $N_c$ are the **total** treatment and control group sizes (not just within the top-$\phi$). The key difference from the cumulative gain: the denominators are the global group sizes $N_t, N_c$, not the local counts $n_t(\phi), n_c(\phi)$.

**What this measures.** At fraction $\phi$, $\frac{n_{t,1}(\phi)}{N_t}$ is the cumulative positive outcome rate contributed by the top-$\phi$ treated users, normalized by the *total* treatment group size. Similarly for control. The Qini value is their difference — it tracks how incremental positive outcomes accumulate as you work down the ranking, with the global normalization ensuring stable estimates even when local treatment/control counts are unequal. At $\phi = 1$, the Qini curve equals $\frac{N_{t,1}}{N_t} - \frac{N_{c,1}}{N_c}$, which is the overall ATE estimated from the test data.

**Random baseline.** A random ranking produces a straight line from $(0, 0)$ to:

$$
\left(1,\; \frac{N_{t,1}}{N_t} - \frac{N_{c,1}}{N_c}\right)
$$

which is the overall ATE estimated from the test data.

### A small worked example

Consider 8 users (4 treated, 4 control) sorted by predicted uplift. The $Y$ column is the observed outcome:

| Rank | User | $\hat{\tau}$ | Treatment | $Y$ |
|:----:|:----:|:----------:|:---------:|:---:|
| 1 | A | 0.9 | 1 | 1 |
| 2 | B | 0.8 | 0 | 0 |
| 3 | C | 0.6 | 1 | 1 |
| 4 | D | 0.5 | 0 | 0 |
| 5 | E | 0.3 | 0 | 1 |
| 6 | F | 0.2 | 1 | 0 |
| 7 | G | 0.1 | 0 | 1 |
| 8 | H | 0.0 | 1 | 0 |

Global: $N_t = 4$, $N_c = 4$, $N_{t,1} = 2$, $N_{c,1} = 2$.

Walking down the ranking, the Qini curve at each step $\phi = i/8$:

| $\phi$ | $n_{t,1}$ | $n_{c,1}$ | Qini = $\frac{n_{t,1}}{4} - \frac{n_{c,1}}{4}$ |
|:------:|:---------:|:---------:|:---:|
| 0 | 0 | 0 | 0 |
| 1/8 | 1 | 0 | 0.25 |
| 2/8 | 1 | 0 | 0.25 |
| 3/8 | 2 | 0 | 0.50 |
| 4/8 | 2 | 0 | 0.50 |
| 5/8 | 2 | 1 | 0.25 |
| 6/8 | 2 | 1 | 0.25 |
| 7/8 | 2 | 2 | 0.00 |
| 8/8 | 2 | 2 | 0.00 |

The curve rises steeply (the model correctly ranks the two treated-positive users at the top), then falls as control-positive users enter. The endpoint is $\frac{2}{4} - \frac{2}{4} = 0$, meaning there is zero overall ATE in this tiny dataset. The random baseline is a flat line at 0. The Qini coefficient $Q$ is the area under the model's curve minus the area under the random line — in this case, the model-curve area is positive, showing it successfully concentrates uplift at the top of the ranking even though the overall ATE is zero.

### The Qini coefficient

The **Qini coefficient** $Q$ is the area between the model's Qini curve and the random baseline:

$$
Q = \int_0^1 \text{Qini}_{\text{model}}(\phi)\, d\phi - \int_0^1 \text{Qini}_{\text{random}}(\phi)\, d\phi
$$

In practice, the integrals are computed as Riemann sums over the $M$ data points:

$$
Q = \sum_{i=0}^{M-1} \frac{1}{2}\left(\text{Qini}(\phi_{i+1}) + \text{Qini}(\phi_i)\right) \cdot \frac{1}{M} \;-\; \frac{1}{2}\left(\frac{N_{t,1}}{N_t} - \frac{N_{c,1}}{N_c}\right)
$$

A higher $Q$ means the model is better at ranking users by true uplift. $Q = 0$ means the model ranks no better than random; the theoretical maximum depends on the data.

### Normalized Qini (q1, q2)

The raw Qini coefficient $Q$ depends on the overall response rate and treatment effect magnitude — a dataset with a large ATE will mechanically produce larger $Q$ values. To compare across datasets or experiments, normalize:

- **$q_1$ (theoretical max):** $Q / Q_{\text{perfect}}$, where $Q_{\text{perfect}}$ is the Qini coefficient of a model that sorts users by their true ITE. This is knowable only in simulation (where the true ITE is known).
- **$q_2$ (practical max):** $Q / Q_{\text{practical}}$, where $Q_{\text{practical}}$ assumes the best achievable sorting given that we only observe one potential outcome per user. This is a more realistic upper bound.

Both $q_1$ and $q_2$ range from 0 to 1, where 1 means the model captures all achievable uplift.

---

## 4. Qini vs. cumulative gain: when to use which

Both curves measure ranking quality, but they differ in how they handle imbalanced treatment/control groups and how they weight different parts of the population:

| Property | Cumulative gain | Qini |
|----------|:-:|:-:|
| Denominators | Local: $n_t(\phi), n_c(\phi)$ | Global: $N_t, N_c$ |
| Sensitive to treatment balance | Yes — if top-$\phi$ users are disproportionately treatment or control, the rate estimate is noisy | Less so — global normalization smooths this |
| Interpretation | "Total incremental outcomes from targeting top $\phi$" | "Cumulative uplift contribution from top $\phi$, globally normalized" |
| Standard in literature | Gutierrez & Gérardy 2017 | Radcliffe 2007, widely used |

**Practical recommendation.** In a randomized experiment with balanced arms (like our Phase 1), the two curves tell nearly the same story. Use both as a sanity check. If they diverge, investigate treatment/control balance in the top-ranked segments.

Both curves produce *plots*. For model selection and reporting, we often need a single number. That's what AUUC and Uplift@k provide.

---

## 5. AUUC: the summary statistic

The **Area Under the Uplift Curve (AUUC)** is the single-number summary most commonly reported. It can be computed for either the cumulative gain or the Qini curve:

$$
\text{AUUC} = \int_0^1 \text{UpliftCurve}(\phi)\, d\phi
$$

By itself, the AUUC isn't very interpretable — its magnitude depends on the overall ATE and sample size. What matters is the comparison:

- **AUUC(model) vs. AUUC(random):** The gap is the model's value-add. If these are close, the model isn't doing much.
- **AUUC(model A) vs. AUUC(model B):** For model selection — pick the model with higher AUUC.
- **AUUC(model) vs. AUUC(perfect):** How much room for improvement remains.

### Uplift@k: a more actionable variant

In practice, you often care about performance at a specific targeting fraction, not across the whole curve. **Uplift@k** measures the observed uplift among the top-$k$% of users ranked by predicted CATE:

$$
\text{Uplift@}k = \frac{n_{t,1}(k\%)}{n_t(k\%)} - \frac{n_{c,1}(k\%)}{n_c(k\%)}
$$

This directly answers: *"If I only had budget to target $k$% of users, what incremental lift would the model generate?"* For Shop Cash, if the budget covers ~20% of the user base, Uplift@20 is the most relevant metric.

AUUC and Uplift@k tell you whether the model *ranks* users well. But the allocation optimizer (see [Objective Design](objective_design.md)) doesn't just use rankings — it uses the CATE *magnitudes* to solve the knapsack. If the magnitudes are wrong, the allocator makes bad decisions even if the ranking is perfect. This is where calibration comes in.

---

## 6. Calibration

Ranking metrics tell you whether the model *orders* users correctly but nothing about whether the CATE *magnitudes* are accurate. A model that predicts $\hat{\tau} = 100$ for everyone with true $\tau = 10$ would have a perfect uplift curve (if the ranking is preserved) but terrible calibration. Calibration matters for two reasons:

1. **The allocation optimizer uses CATE magnitudes.** The knapsack formulation maximizes $\sum_u [\hat{\tau}(u,t) - c(t)]$. If $\hat{\tau}$ is systematically too high, the optimizer will over-allocate rewards (the net uplift will be less than projected). If too low, it will under-allocate.

2. **Stakeholder trust.** If you tell the business "this user will generate $15 of incremental GMV from a $5 reward" and the actual lift is $3, the model loses credibility — even if the *ranking* was correct.

### How to assess calibration

**Binned calibration plot.** Group users into deciles by predicted CATE $\hat{\tau}$. Within each decile, compute the observed uplift (mean outcome in treatment minus mean outcome in control). Plot predicted vs. observed:

- A perfectly calibrated model produces points on the 45° line.
- Points above the line: the model under-predicts uplift.
- Points below: the model over-predicts.

$$
\text{For decile } d: \quad \hat{\tau}_d = \text{mean}(\hat{\tau}_i : i \in d), \quad \tau_d^{\text{obs}} = \bar{Y}_{\text{treated}, d} - \bar{Y}_{\text{control}, d}
$$

**Why this works despite the fundamental problem.** We can't observe individual treatment effects, but within a group of users, the difference in mean outcomes between treated and control is an unbiased estimate of the group-level CATE (by the same randomization argument from [Note 1](causal_inference_uplift_modeling.md)). Larger groups give more precise estimates, which is why we bin into deciles rather than evaluating calibration per-user.

[FIG:ORIGINAL — calibration plot for an uplift model showing predicted CATE (x-axis) vs. observed uplift (y-axis) with decile bins, the 45-degree perfect calibration line, and error bars on the observed uplift estimates]

**Recalibration.** If the model is well-ranked but poorly calibrated (common with boosted tree models that shrink predictions), a simple post-hoc fix is **isotonic regression** of $\hat{\tau}$ against the transformed outcome $Y^*$. Isotonic regression fits a monotonically non-decreasing step function, which by construction preserves the ranking while adjusting the scale to match observed uplift levels.

Everything above evaluates the model on held-out experimental data — essentially asking "would this model have worked well on the Phase 1 population?" The definitive test is whether the model actually *generates more value in production* than naive allocation. This requires a new experiment.

---

## 7. Evaluating the system: Phase 2 experiment design

### Phase 2 design: three arms

| Arm | Description | What it measures |
|-----|-------------|-----------------|
| **Control** | No reward | Baseline: $\text{E}[Y \mid T = 0]$ |
| **Random** | Reward from {$5, $10, $15, $20} assigned uniformly at random | The value of giving rewards without targeting: $\text{E}[Y \mid \text{random reward}] - \text{E}[Y \mid T = 0]$ |
| **Optimized** | Reward tier assigned by the uplift model + knapsack allocator | The value of *targeted* rewards: $\text{E}[Y \mid \text{optimized reward}] - \text{E}[Y \mid T = 0]$ |

### What to compare

The key comparisons, all of which should be evaluated as **intent-to-treat** differences (compare arm means, ignoring whether users actually redeemed):

1. **Optimized vs. Control:** The total incremental GMV from the model-driven system. This is the top-line number the business cares about.

2. **Optimized vs. Random:** The incremental value of *targeting*. This is the cleanest test of the model. **Critical design requirement:** the total reward budget must be equalized between the Random and Optimized arms. If the Optimized arm spends less (by not rewarding inelastic users), the comparison conflates targeting quality with budget savings. Equalizing spend isolates the pure targeting effect: same dollars in, more incremental GMV out.

3. **Random vs. Control:** The average effect of giving rewards without any targeting. This is essentially the ATE from Phase 1, reconfirmed in Phase 2.

### Metrics to report

For each comparison, compute:

- **Incremental GMV** (the primary outcome): mean GMV difference between arms, with confidence intervals.
- **Net incremental value**: incremental GMV minus reward cost. This is what the business actually captures.
- **ROIS**: incremental GMV per dollar of reward spend. The optimized arm should have higher ROIS than the random arm (same or more lift, same or less spend).
- **Budget utilization**: what fraction of the budget was actually spent in each arm, and how was it distributed across tiers. The optimized arm may concentrate spend on fewer, higher-value users.

### Sample size considerations

The Phase 2 experiment is comparing *systems*, not individual treatment effects. The relevant effect size is the difference between the optimized and random arms, which may be smaller than the difference between any single arm and control. Power analysis should target the **Optimized vs. Random** comparison, since this is the test of the model's value. If the per-user effect of targeting is, say, $2 of incremental GMV, and you want to detect this with 80% power at $\alpha = 0.05$, you'll need substantially more users per arm than for the Optimized vs. Control comparison (where the effect is larger).

---

## 8. Putting it all together: the evaluation playbook

| Stage | What you evaluate | Metric | Data source |
|-------|-------------------|--------|-------------|
| **Model development** | Does the model rank users correctly? | Uplift curve, AUUC, Qini | Held-out Phase 1 data (cross-validation) |
| **Model selection** | Which model is best? | AUUC comparison across S/T/X-learner, causal forest | Held-out Phase 1 data |
| **Calibration check** | Are CATE magnitudes accurate? | Binned calibration plot, predicted vs. observed | Held-out Phase 1 data |
| **Allocation sanity check** | Does the optimizer produce reasonable assignments? | Distribution of reward tiers, total spend vs. budget | Model predictions on target population |
| **System evaluation** | Does targeting outperform random? | Optimized vs. Random GMV, net value, ROIS | Phase 2 experiment |

The first four stages use Phase 1 data and happen *before* the Phase 2 experiment launches. They give you confidence (or flag problems) before committing production budget. The fifth stage is the definitive test.

**The evaluation stack, in one sentence.** Uplift curves tell you the model *ranks* well, calibration tells you the *magnitudes* are right, and the Phase 2 experiment tells you the whole system *works in production*.

---

## Practical implementation

The following code computes the Qini curve and AUUC for a binary treatment uplift model, using only numpy and pandas. This bridges the gap between the formulas above and the actual computation — the tricky part is correctly sorting by predicted uplift and cumulatively tracking treatment/control outcomes.

```python
import numpy as np
import pandas as pd

def qini_curve(y_true, uplift_pred, treatment):
    """
    Compute the Qini curve.
    
    Args:
        y_true: observed outcomes (GMV or binary conversion)
        uplift_pred: predicted CATE scores (higher = more uplift)
        treatment: binary treatment indicator (1 = treated, 0 = control)
    
    Returns:
        fractions: array of targeting fractions (0 to 1)
        qini_values: Qini curve values at each fraction
    """
    y_true = np.asarray(y_true, dtype=float)
    uplift_pred = np.asarray(uplift_pred, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    
    order = np.argsort(-uplift_pred)
    y_sorted = y_true[order]
    t_sorted = treatment[order]
    
    N_t = treatment.sum()
    N_c = len(treatment) - N_t
    
    cum_t_outcomes = np.cumsum(y_sorted * t_sorted)
    cum_c_outcomes = np.cumsum(y_sorted * (1 - t_sorted))
    
    qini_values = cum_t_outcomes / N_t - cum_c_outcomes / N_c
    fractions = np.arange(1, len(y_true) + 1) / len(y_true)
    
    qini_values = np.insert(qini_values, 0, 0)
    fractions = np.insert(fractions, 0, 0)
    
    return fractions, qini_values

def qini_auc(y_true, uplift_pred, treatment):
    """Compute the area between the model's Qini curve and the random baseline."""
    fractions, qini_values = qini_curve(y_true, uplift_pred, treatment)
    
    model_area = np.trapz(qini_values, fractions)
    random_area = 0.5 * qini_values[-1]  # triangle under the random line
    
    return model_area - random_area
```

**Reading the output.** `qini_auc` returns a single number: the area between the model's Qini curve and the random baseline. Positive means the model ranks better than random; larger is better. Use this to compare models (T-learner vs. causal forest) on held-out Phase 1 data.

**Multi-treatment extension.** The code above assumes binary treatment (treated/control). For the Shop Cash setup with 4 reward tiers, you have two options. First, compute a separate Qini curve for each tier vs. control (collapsing the other tiers out of the analysis). This gives tier-specific ranking quality. Second — and more relevant for the allocator — evaluate the *system-level CATE*: for each user, the model + allocator selects a single optimal tier, producing a single predicted uplift $\hat{\tau}^*(x)$. You can then compute a single Qini curve where "treated" means "assigned any non-zero reward by the allocator" and the predicted uplift is $\hat{\tau}^*(x)$. This evaluates the full pipeline, not just a single tier model.

---

## Sources and Further Reading

- **Radcliffe, N. J.** (2007). "Using Control Groups to Target on Predicted Lift: Building and Assessing Uplift Models." *Direct Marketing Analytics Journal*, 1, 14–21. — Introduces the Qini curve and coefficient.

- **Gutierrez, P. & Gérardy, J. Y.** (2017). "Causal Inference and Uplift Modelling: A Review of the Literature." *JMLR Workshop and Conference Proceedings*, 67, 1–13. — Defines the cumulative gain chart and provides a comprehensive survey of uplift evaluation methods.

- **Athey, S. & Imbens, G. W.** (2015). "Machine Learning Methods for Estimating Heterogeneous Causal Effects." *stat*, 1050(5). — Introduces the transformed outcome approach.

- **Diemert, E., Betlei, A., Renaudin, C., & Amini, M. R.** (2021). "A Large Scale Benchmark for Uplift Modeling." *Proceedings of the KDD Workshop on Causal Discovery*. — Provides benchmarks and practical guidance for uplift evaluation.

- **scikit-uplift** documentation: [uplift-modeling.com](https://www.uplift-modeling.com/) — Python implementation of AUUC, Qini coefficient, Uplift@k, and other metrics.

- **pylift** documentation: [pylift.readthedocs.io](https://pylift.readthedocs.io/) — Detailed mathematical treatment of Qini curves and the transformed outcome method.
