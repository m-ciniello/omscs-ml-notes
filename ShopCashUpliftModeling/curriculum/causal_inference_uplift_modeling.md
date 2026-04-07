# Causal Inference & Uplift Modeling

This note covers the theoretical foundations and practical methods for estimating **heterogeneous treatment effects** — the core ML problem behind Shop Cash reward optimization. It is organized in two parts. Part I builds the causal inference framework: what it means to ask "how much more would this user spend if given a $10 reward?", why that question is fundamentally different from a prediction problem, and what assumptions let us answer it from experimental data. Part II introduces the ML methods — meta-learners (S, T, X-learner), causal forests — that estimate these effects at the individual user level, then extends them to the multi-treatment setting (4 reward tiers + control) that our problem requires.

The thread connecting both parts: **we are not predicting outcomes, we are estimating the causal effect of an intervention.** Every modeling choice in Part II is shaped by this distinction.

---

## Part I — Causal Inference Foundations

This section introduces the potential outcomes framework, defines the estimands we care about (ATE, CATE), explains why the individual treatment effect is fundamentally unobservable, and shows how randomization solves the identification problem. The goal is to build enough formal scaffolding that the modeling decisions in Part II feel motivated rather than arbitrary.

---

### 1.1 The causal question behind reward optimization

Standard ML asks predictive questions: *given a user's features, what will they spend?* The reward optimization problem asks a different kind of question:

> **If I give this user a $10 reward, how much *more* will they spend compared to getting nothing?**

This is a **causal** question — it asks about the effect of an intervention, not about a prediction. The distinction matters because:

- A user with high predicted GMV is not necessarily a good reward target. They might spend $200 regardless of whether they receive a reward (User 1 in our [counterfactual examples](../uplift_modeling/objective_design.md)). Giving them $10 generates $0 in incremental GMV.
- A user with low predicted GMV might be an excellent target if a reward would shift their behavior significantly (User 6 in our examples: $0 baseline GMV, $70 incremental GMV from a $5 reward).

**The relevant quantity is not $\text{E}[Y \mid X]$ (what will they spend?) but the causal effect of the reward on spending.** We'll formalize this precisely in §1.2 as the difference in *potential outcomes* — what a user would spend *with* the reward minus what they would spend *without* it. Estimating this causal quantity is the subject of the rest of this note.

---

### 1.2 Potential outcomes and the Rubin causal model

The **potential outcomes framework** (Neyman 1923, Rubin 1974) formalizes causal questions by imagining all possible outcomes for a user, one for each treatment they *could* receive — regardless of which they actually receive.

**Setup.** For each user $u$ and each treatment level $t \in \{0, 5, 10, 15, 20\}$, define the **potential outcome**:

$$
Y_u(t) = \text{the GMV user } u \text{ would generate if assigned treatment } t
$$

These are fixed (non-random) quantities — they represent the user's latent response to each possible intervention. The full set $\{Y_u(0), Y_u(5), Y_u(10), Y_u(15), Y_u(20)\}$ is called the user's **potential outcome vector**.

The counterfactual table from the objective design doc is exactly this — each row is a potential outcome:

| User | $Y_u(0)$ | $Y_u(5)$ | $Y_u(10)$ |
|:----:|:--------:|:---------:|:----------:|
| 1 | $25 | $25 | $25 |
| 2 | $25 | $30 | $50 |
| 3 | $25 | $55 | $80 |

**The individual treatment effect (ITE)** is the causal effect of treatment $t$ versus control for user $u$:

$$
\tau_u(t) = Y_u(t) - Y_u(0)
$$

For User 2 at $t = 10$: $\tau_2(10) = 50 - 25 = 25$. This is the quantity the uplift model must estimate.

---

### 1.3 The fundamental problem of causal inference

Here is the core difficulty: **for any given user, we only observe one potential outcome** — the one corresponding to the treatment they actually received.

Define the **observed outcome** as:

$$
Y_u^{\text{obs}} = Y_u(T_u)
$$

where $T_u$ is the treatment actually assigned to user $u$. If User 2 is assigned to the $10 treatment arm, we observe $Y_2^{\text{obs}} = Y_2(10) = 50$. We do *not* observe $Y_2(0) = 25$, $Y_2(5) = 30$, or $Y_2(15)$ or $Y_2(20)$.

This is called the **fundamental problem of causal inference** (Holland 1986): the individual treatment effect $\tau_u(t) = Y_u(t) - Y_u(0)$ requires two quantities, but we can only ever observe one of them. The other is a **counterfactual** — what *would have* happened under a different treatment.

**Why this matters for modeling.** In standard supervised learning, you have $(X_i, Y_i)$ pairs and you learn to predict $Y$ from $X$. In uplift modeling, the "label" you want to predict — the ITE $\tau_u(t)$ — is never directly observed for any user. You cannot simply regress $\tau$ on $X$ because $\tau$ does not appear in your dataset. Every method in Part II is, at its core, a strategy for working around this missing-data problem.

---

### 1.4 Average and conditional treatment effects

Since individual treatment effects are unobservable, we work with population-level summaries.

**The average treatment effect (ATE)** is the expected ITE across the user population:

$$
\text{ATE}(t) = \text{E}[\tau_u(t)] = \text{E}[Y(t) - Y(0)] = \text{E}[Y(t)] - \text{E}[Y(0)]
$$

The ATE answers: *"On average, how much does a $t reward increase GMV?"* This is what a standard A/B test estimates. For example, if we randomize users into control and $10-reward arms and compare mean GMV, the difference in means is an unbiased estimate of $\text{ATE}(10)$.

The ATE is useful for answering "should we run this program at all?" but useless for optimization. It treats all users as interchangeable — it says nothing about *who* should get the reward.

**The conditional average treatment effect (CATE)** conditions on user features $X$:

$$
\tau(x, t) = \text{E}[Y(t) - Y(0) \mid X = x]
$$

The CATE answers: *"For a user with features $x$, how much does a $t reward increase their expected GMV?"* This is the object the uplift model estimates, and it is exactly the $\text{CATE}(u, t)$ that feeds into the allocation optimizer from the [objective design](../uplift_modeling/objective_design.md).

**Why heterogeneity is the whole game.** If the treatment effect were homogeneous — the same for every user — there would be nothing to optimize. You'd either give everyone the reward (if $\text{ATE} > \text{cost}$) or no one. The value of uplift modeling comes entirely from **heterogeneity**: different users respond differently, and the CATE captures this variation. In our examples:

| User | $\tau(x, 5)$ | $\tau(x, 10)$ | Response pattern |
|:----:|:------------:|:--------------:|:-----------------|
| 1 (inelastic) | $0 | $0 | No response to any reward — never target |
| 2 (moderate responder) | $5 | $25 | Responds strongly to $10 but weakly to $5 |
| 3 (high-efficiency) | $30 | $55 | Responds to both, but $5 already captures most of the value |

The ATE of the $10 reward across these three users is $(0 + 25 + 55)/3 \approx \$27$, which suggests the program "works on average." But the ATE masks the real story: User 1 should never receive a reward, User 2 needs the $10 to move, and User 3 is highly responsive even at $5. The optimal policy isn't "give everyone $10" — it's "target *differently* based on the CATE vector." This is why we need CATE estimation, not just an A/B test.

---

### 1.5 Identification: how randomization solves the problem

We said the fundamental problem prevents us from observing individual treatment effects. So how can we estimate the CATE? We need an **identification strategy** — a set of assumptions under which the causal quantity $\tau(x, t)$ can be recovered from observable data.

**The naïve approach fails.** Suppose we have an observational dataset (no randomization) and we estimate:

$$
\hat{\tau}^{\text{naïve}}(t) = \text{E}[Y \mid T = t] - \text{E}[Y \mid T = 0]
$$

This compares the average GMV of users who *happened to receive* treatment $t$ versus those who received control. The problem is **selection bias**: users who receive rewards might systematically differ from those who don't. If the marketing team historically targeted high-value users with larger rewards, then $\text{E}[Y \mid T = 10]$ would be high even if the reward had zero causal effect — simply because high-value users were selected into the treatment group.

We can decompose this formally. By the consistency assumption ($Y = Y(T)$, i.e. the observed outcome equals the potential outcome under the assigned treatment):

$$
\text{E}[Y \mid T = t] - \text{E}[Y \mid T = 0] = \text{E}[Y(t) \mid T = t] - \text{E}[Y(0) \mid T = 0]
$$

Adding and subtracting $\text{E}[Y(0) \mid T = t]$:

$$
= \underbrace{\text{E}[Y(t) - Y(0) \mid T = t]}_{\text{ATT: causal effect on the treated}} + \underbrace{\text{E}[Y(0) \mid T = t] - \text{E}[Y(0) \mid T = 0]}_{\text{selection bias}}
$$

The first term is the **average treatment effect on the treated (ATT)** — the causal effect among those who actually received treatment. The second term is **selection bias**: the difference in *baseline spending* (what they'd spend without any reward) between the treatment and control groups. This term has nothing to do with the treatment effect and everything to do with who was selected. If high-value users were targeted with rewards, $\text{E}[Y(0) \mid T = t]$ is inflated, and the naïve estimator overstates the causal effect.

**Randomization eliminates selection bias.** In a randomized experiment, treatment assignment $T$ is independent of all potential outcomes:

$$
T \perp\!\!\!\perp \{Y(0), Y(5), Y(10), Y(15), Y(20)\}
$$

This is called **unconfoundedness** (or ignorability). When it holds, two things happen:

1. The selection bias term vanishes: $\text{E}[Y(0) \mid T = t] = \text{E}[Y(0)]$ for all $t$, so the bias is zero.
2. The ATT equals the ATE: $\text{E}[Y(t) - Y(0) \mid T = t] = \text{E}[Y(t) - Y(0)]$, since treatment assignment is independent of who benefits.

Together, these mean the simple difference in means is an unbiased estimator of the ATE.

For the CATE, the analogous condition is **conditional unconfoundedness**:

$$
T \perp\!\!\!\perp \{Y(0), Y(5), Y(10), Y(15), Y(20)\} \mid X
$$

In a randomized experiment, this holds automatically (since $T$ is independent of everything, it's certainly independent conditional on $X$).

**Three assumptions for causal identification.** The full set of assumptions we need:

1. **SUTVA (Stable Unit Treatment Value Assumption).** User $u$'s outcome depends only on their own treatment assignment, not on anyone else's. In the Shop Cash context, this means one user's reward doesn't affect another user's spending. This is plausible — users shop independently. (It would be violated if, say, users could transfer rewards or if rewards triggered viral sharing effects.)

2. **Unconfoundedness.** $T \perp\!\!\!\perp \{Y(t)\}_t \mid X$. Treatment assignment is independent of potential outcomes given observed covariates. **Guaranteed by randomization** — this is the whole point of running the Phase 1 experiment.

3. **Overlap (positivity).** $0 < P(T = t \mid X = x) < 1$ for all $t$ and $x$. Every user has a positive probability of receiving every treatment. In a randomized experiment with equal allocation across arms, $P(T = t) = 1/K$ for all users, so overlap holds trivially.

When all three assumptions hold, the CATE is **identified** — it can be expressed purely in terms of observable quantities:

$$
\tau(x, t) = \text{E}[Y(t) - Y(0) \mid X = x] = \text{E}[Y \mid X = x, T = t] - \text{E}[Y \mid X = x, T = 0]
$$

The left side involves unobservable potential outcomes; the right side involves only observable data $(X, T, Y^{\text{obs}})$. This identity is the bridge between the causal estimand and the statistical estimation problem — and it is what makes the meta-learners in Part II possible.

---

### Part I wrap-up

The potential outcomes framework gives us a precise language for what we're estimating: the CATE $\tau(x, t) = \text{E}[Y(t) - Y(0) \mid X = x]$, the expected incremental GMV for a user with features $x$ under treatment tier $t$. The fundamental problem — we only observe one potential outcome per user — means we can never directly observe this quantity. But randomization (our Phase 1 experiment) guarantees the identification assumptions (unconfoundedness, overlap), which lets us express the CATE in terms of observable data: $\text{E}[Y \mid X, T = t] - \text{E}[Y \mid X, T = 0]$.

Part II takes this identification result and asks: how do we actually *estimate* $\tau(x, t)$ from finite experimental data? Each method makes different tradeoffs between simplicity, flexibility, statistical efficiency, and the ability to quantify uncertainty.

---

## Part II — Uplift Modeling (CATE Estimation)

With identification established, the problem reduces to: given experimental data $\{(X_i, T_i, Y_i)\}_{i=1}^n$, estimate the function $\tau(x, t) = \text{E}[Y \mid X = x, T = t] - \text{E}[Y \mid X = x, T = 0]$. This section covers four families of estimators — the S-learner, T-learner, X-learner, and causal forests — each making different tradeoffs. We first develop the methods for **binary treatment** (treated vs. control), where the ideas are cleanest, then extend to the multi-treatment case in §2.6.

Throughout, let $\mu_t(x) = \text{E}[Y \mid X = x, T = t]$ denote the conditional response surface under treatment $t$. The CATE for binary treatment is $\tau(x) = \mu_1(x) - \mu_0(x)$.

---

### 2.1 Why this isn't standard supervised learning

It's tempting to think of CATE estimation as "just another regression problem." The key obstacle: **the label $\tau_i$ is never observed.** In a standard supervised learning setup, we have input-output pairs $(X_i, Y_i)$ and fit a function $f(X) \approx Y$. In uplift modeling, the quantity we want to predict — $\tau_i = Y_i(1) - Y_i(0)$ — does not appear in the dataset. Each user contributes either $Y_i(1)$ (if treated) or $Y_i(0)$ (if control), never both.

This rules out a direct regression approach. Every method below is an indirect strategy — a way to recover $\tau(x)$ by estimating other quantities (response surfaces, imputed treatment effects, or adaptive partitions) from which the CATE can be derived.

---

### 2.2 The S-learner

**Idea.** Fit a single model that predicts the outcome $Y$ as a function of both user features $X$ and the treatment indicator $T$, then compute the CATE by differencing predictions:

$$
\hat{\mu}(x, t) = \hat{f}(x, t) \qquad \hat{\tau}^S(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)
$$

The "S" stands for "single" — one model handles both treated and control units.

**In practice:** concatenate $T$ as an additional feature to $X$, train any off-the-shelf regressor (gradient boosted trees, neural network, etc.) on the full dataset $\{(X_i, T_i, Y_i)\}$, then predict with $T = 1$ and $T = 0$ and take the difference.

**What it gets right.** Simplicity: no special causal machinery, just a standard regression with one extra input feature. It uses all the data in a single model, so it's sample-efficient.

**Where it breaks down: regularization bias.** The treatment indicator $T$ is a single binary feature competing with potentially hundreds of user features. Regularized models (L1/L2 regression, tree-based methods with pruning) will shrink the effect of $T$ toward zero — especially if the treatment effect is small relative to the baseline variation in $Y$.

Consider a concrete example: suppose $Y = 100 + 0.5T + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 10^2)$. The true CATE is $0.5$ everywhere. A regularized model fitting $Y \sim X + T$ will heavily shrink the coefficient on $T$ because the signal-to-noise ratio for the treatment effect is $0.5/10 = 0.05$ — tiny. The S-learner would estimate $\hat{\tau} \approx 0$, missing the effect entirely.

**The deeper issue.** The S-learner is estimating the full response surface $\mu(x, t)$ and then *differencing* to get the CATE. This means it needs to be accurate at the level of the *difference* $\mu(x,1) - \mu(x,0)$, not just at the level of $\mu$ itself. If $\mu(x,0) = 500$ and $\mu(x,1) = 510$, the S-learner needs to nail the difference between 500 and 510 — a 2% relative difference — to recover a CATE of 10. Any model error that is correlated across $T = 0$ and $T = 1$ will wash out, but error that differs across them will contaminate the CATE.

**When to use it.** The S-learner works well when (a) treatment effects are large relative to baseline variance, or (b) you use a flexible model (like a deep net or unregularized boosted trees) that won't shrink the treatment variable. It's a reasonable baseline but rarely the best choice.

![Data flow comparison of the three meta-learner architectures. The S-learner fits a single model and differences predictions; the T-learner fits separate treatment/control models; the X-learner adds a cross-imputation stage with propensity-weighted combination.](images/meta_learner_architectures.png)
*Data flow comparison of S-learner, T-learner, and X-learner architectures. Orange = treatment data/models, blue = control data/models.*

The S-learner's core problem is that it treats the treatment effect as "just another signal" in a single model, where it gets drowned out. The natural question: what if we don't force treatment and control into the same model at all?

---

### 2.3 The T-learner

**Idea.** Fit *separate* outcome models for the treated and control groups, then difference the predictions:

$$
\hat{\mu}_0(x) = \hat{f}_0(x) \quad \text{(fit on control data only)} \qquad \hat{\mu}_1(x) = \hat{f}_1(x) \quad \text{(fit on treated data only)}
$$

$$
\hat{\tau}^T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
$$

The "T" stands for "two" — two separate models.

**What it gets right.** Each model is free to learn an arbitrarily different response surface for treated vs. control users. There is no regularization bias against finding treatment effects, because the treatment indicator doesn't compete with other features — it's baked into the model structure.

**Where it breaks down: doubled variance and no shared structure.**

1. **Data splitting.** Each model sees only the data from its arm. In a 50/50 randomized experiment, each model uses $n/2$ observations. With $K = 5$ arms (our setting), the control model sees $n/5$ observations and each treatment model sees $n/5$. The CATE estimate $\hat{\tau}^T(x) = \hat{\mu}_t(x) - \hat{\mu}_0(x)$ inherits variance from both models:

$$
\text{Var}[\hat{\tau}^T(x)] = \text{Var}[\hat{\mu}_t(x)] + \text{Var}[\hat{\mu}_0(x)]
$$

This is the price of flexibility — by refusing to share any structure between treatment and control, the T-learner wastes statistical power.

2. **No structure sharing.** In many real problems, the response surfaces $\mu_0(x)$ and $\mu_1(x)$ are *mostly similar* — the treatment effect is a small perturbation on top of a large baseline. The T-learner ignores this, fitting two models from scratch. The S-learner is at the opposite extreme (fully shared). The X-learner tries to get the best of both worlds.

**When to use it.** The T-learner is a good default when (a) you have enough data per arm, and (b) the treatment and control response surfaces might differ in complex ways that a shared model would miss. With the rich user features available (full purchase history, click behavior, etc.) and a large Phase 1 sample, it's often competitive.

The S-learner and T-learner sit at opposite ends of a spectrum: the S-learner fully shares structure (one model, risking bias), while the T-learner shares nothing (separate models, risking variance). Can we get the benefits of both — use the full data while still allowing flexible treatment effects?

---

### 2.4 The X-learner

The X-learner (Künzel et al. 2019) is a two-stage method designed to improve on the T-learner by using *imputed* individual treatment effects as pseudo-labels. It is especially effective when the treatment and control groups are imbalanced (one much larger than the other), but it can help even with balanced groups.

**Motivation.** The T-learner estimates CATE as $\hat{\mu}_1(x) - \hat{\mu}_0(x)$, where each model is fit on half the data. Can we do better? The X-learner's insight: once we have initial estimates $\hat{\mu}_0$ and $\hat{\mu}_1$, we can *impute* the missing potential outcome for each unit and then directly regress the imputed treatment effect on features.

**Stage 1 — fit response surfaces** (same as the T-learner):

$$
\hat{\mu}_0(x) \text{ fit on } \{(X_i, Y_i) : T_i = 0\} \qquad \hat{\mu}_1(x) \text{ fit on } \{(X_i, Y_i) : T_i = 1\}
$$

**Stage 2 — impute individual treatment effects.** For each unit, compute a pseudo-label $\tilde{D}_i$ by filling in the unobserved potential outcome with the Stage 1 estimate:

- For **treated** units ($T_i = 1$): the observed outcome is $Y_i = Y_i(1)$, and the missing counterfactual $Y_i(0)$ is estimated by $\hat{\mu}_0(X_i)$:

$$
\tilde{D}_i^1 = Y_i - \hat{\mu}_0(X_i) \approx Y_i(1) - Y_i(0) = \tau_i
$$

- For **control** units ($T_i = 0$): the observed outcome is $Y_i = Y_i(0)$, and the missing counterfactual $Y_i(1)$ is estimated by $\hat{\mu}_1(X_i)$:

$$
\tilde{D}_i^0 = \hat{\mu}_1(X_i) - Y_i \approx Y_i(1) - Y_i(0) = \tau_i
$$

Now we have pseudo-labels $\tilde{D}_i$ that approximate the individual treatment effect — one for every unit in the dataset, not just within each arm. We then fit two CATE models:

$$
\hat{\tau}_1(x) \text{ fit on } \{(X_i, \tilde{D}_i^1) : T_i = 1\} \qquad \hat{\tau}_0(x) \text{ fit on } \{(X_i, \tilde{D}_i^0) : T_i = 0\}
$$

**Stage 3 — combine with propensity weighting:**

$$
\hat{\tau}^X(x) = g(x) \cdot \hat{\tau}_0(x) + (1 - g(x)) \cdot \hat{\tau}_1(x)
$$

where $g(x) = P(T = 1 \mid X = x)$ is the **propensity score**. In a randomized experiment with equal allocation, $g(x) = 0.5$ for all $x$, so this is a simple average.

**What does this accomplish?** Compare with the T-learner. The T-learner estimates $\hat{\tau}^T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$ — it differences two outcome predictions, each fit on half the data. The X-learner instead creates *pseudo-labels that directly approximate the ITE*, then regresses on those. The key advantage: it uses information from *both* groups to construct each pseudo-label.

Notice which imputation uses which model:

- $\tilde{D}_i^1$ (for treated units) uses $\hat{\mu}_0$ — the **control** model. If the control group is large, $\hat{\mu}_0$ is accurate, so the imputed effects for treated units are reliable.
- $\tilde{D}_i^0$ (for control units) uses $\hat{\mu}_1$ — the **treatment** model. If the treatment group is large, $\hat{\mu}_1$ is accurate, so the imputed effects for control units are reliable.

Each imputation "borrows strength" from the opposite arm. The propensity-weighted combination then puts more weight on the CATE estimate derived from the *larger* group — the group whose imputed pseudo-labels are more reliable. This is why the X-learner excels with imbalanced groups: when treatment is rare, the control group is large → $\hat{\mu}_0$ is accurate → the pseudo-labels $\tilde{D}_i^1$ for treated units are good → $\hat{\tau}_1$ captures the CATE well → and $g(x)$ is small, so $\hat{\tau}_1$ gets most of the weight.

**When to use it.** The X-learner consistently outperforms the T-learner when (a) groups are imbalanced, or (b) the response surfaces are smooth and the Stage 1 estimates are accurate. It's a strong default for binary treatment. The main downside is complexity: three stages, two imputation steps, and a propensity weighting step — more moving parts than the T-learner.

All three meta-learners share a common structure: fit outcome models $\hat{\mu}_t(x)$, then derive the CATE from them. This means they inherit whatever biases and errors those outcome models have. Causal forests take a fundamentally different approach — they estimate the CATE *directly*, without ever fitting an outcome model.

---

### 2.5 Causal forests

Causal forests (Athey & Imbens 2016; Wager & Athey 2018), implemented in the **Generalized Random Forests (GRF)** framework, take a fundamentally different approach. Rather than fitting outcome models and differencing, they directly partition the feature space to find regions where the treatment effect varies.

**Core idea.** A standard random forest splits the feature space to minimize prediction error for $Y$. A causal forest splits to maximize **heterogeneity in the treatment effect** across the two sides of each split. Regions where treated and control outcomes diverge most are split first; regions where the treatment effect is homogeneous are left unsplit.

**How a single tree works.** At each candidate split (feature $j$, threshold $s$), the tree evaluates: *does splitting here produce child nodes with meaningfully different CATEs?* The intuition is captured by a criterion that maximizes:

$$
n_L \cdot n_R \cdot (\hat{\tau}_L - \hat{\tau}_R)^2
$$

where $\hat{\tau}_L, \hat{\tau}_R$ are the estimated CATEs in the left and right children and $n_L, n_R$ are their sample sizes. This is analogous to how a standard regression tree maximizes variance reduction, but applied to the treatment effect rather than the outcome. (The actual GRF implementation uses a more sophisticated gradient-based criterion that accounts for estimation uncertainty, but the core idea — split where treatment effects diverge most — is the same.)

Within each leaf, the CATE is estimated by the difference in mean outcomes between treated and control units:

$$
\hat{\tau}_{\text{leaf}} = \bar{Y}_{\text{treated}} - \bar{Y}_{\text{control}}
$$

**Honesty.** A critical innovation in causal forests is **honesty**: the data used to determine the tree structure (where to split) is different from the data used to estimate the leaf-level CATEs. This is analogous to sample splitting in statistics — it prevents overfitting the CATE estimates to the same data that defined the partitions. Concretely, each tree uses one subsample to find splits and a separate subsample to populate the leaves.

**Why honesty matters.** Without it, the tree would find splits that happen to separate high- and low-outcome users (not high- and low-treatment-effect users) due to noise, and then estimate biased CATEs in the resulting leaves. Honesty ensures that the CATE estimates are unbiased, at the cost of slightly less adaptive partitioning.

**The forest.** As with standard random forests, many honest causal trees are averaged. Each tree is grown on a random **subsample** (drawn *without* replacement — not bootstrap) with random feature selection at each split. Subsampling without replacement is a deliberate choice: it is required for the asymptotic normality result that gives causal forests their confidence intervals (Wager & Athey 2018). The forest-level CATE estimate is:

$$
\hat{\tau}^{CF}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{\tau}_b(x)
$$

where $\hat{\tau}_b(x)$ is the CATE from tree $b$ evaluated at $x$.

**Confidence intervals — a major advantage.** Unlike the meta-learners, causal forests provide asymptotically valid confidence intervals for $\hat{\tau}(x)$. Under regularity conditions, the forest CATE estimate is asymptotically normal:

$$
\frac{\hat{\tau}^{CF}(x) - \tau(x)}{\hat{\sigma}(x)} \xrightarrow{d} \mathcal{N}(0, 1)
$$

where $\hat{\sigma}(x)$ is a variance estimate computed from the forest. This means we can construct a 95% confidence interval $\hat{\tau}(x) \pm 1.96 \cdot \hat{\sigma}(x)$ for the CATE at any point $x$.

**Why this matters for reward optimization.** Confidence intervals are directly useful for the allocation step. When the CATE point estimate is high but the confidence interval is wide, we are uncertain about whether the reward will generate incremental GMV. A risk-averse allocator might prefer a user with a slightly lower point estimate but much tighter interval. This connects to the uncertainty-aware allocation discussed in the [curriculum](curriculum.md) — and it's something meta-learners don't provide out of the box.

![An uplift tree from causalml. Each node shows the split condition, sample sizes (treatment and control), the uplift score (estimated CATE), and a p-value. Leaf colors indicate positive (blue) vs. negative (green) uplift. The bottom-left leaf has the highest uplift (0.12, p=0.0008), while the top-right leaf has negative uplift (−0.09). Source: Uber causalml documentation.](images/uplift_tree_causalml.png)
*An uplift tree trained on synthetic data. Each node displays the split condition, treatment/control sample sizes, uplift score (CATE), and statistical significance. Blue leaves have positive uplift (persuadables); green leaves have negative uplift (sleeping dogs). Source: causalml (Uber).*

---

### 2.6 Multi-treatment extensions

Everything above assumed binary treatment (treated vs. control). The Shop Cash problem has **5 arms**: control + four reward tiers ($5, $10, $15, $20). This section describes how each method extends.

The target estimand becomes a **vector of CATEs** for each user:

$$
\boldsymbol{\tau}(x) = \bigl(\tau(x, 5),\; \tau(x, 10),\; \tau(x, 15),\; \tau(x, 20)\bigr)
$$

where $\tau(x, t) = \text{E}[Y(t) - Y(0) \mid X = x]$. The allocation optimizer chooses $t^*(x) = \arg\max_t \bigl[\tau(x,t) - c(t)\bigr]$ subject to the budget constraint.

#### S-learner: natural extension

Include treatment as a categorical feature (or a set of dummy variables / the dollar amount itself). Predict $\hat{\mu}(x, t)$ for each tier and difference against $\hat{\mu}(x, 0)$:

$$
\hat{\tau}^S(x, t) = \hat{\mu}(x, t) - \hat{\mu}(x, 0)
$$

The regularization bias issue from §2.2 applies with equal force — now the treatment features are a handful of dummies competing with hundreds of user features.

#### T-learner: $K + 1$ separate models

Fit a separate model for each arm: $\hat{\mu}_0, \hat{\mu}_5, \hat{\mu}_{10}, \hat{\mu}_{15}, \hat{\mu}_{20}$. Each model sees only $n / (K+1)$ observations (roughly $n/5$ with equal allocation). The CATE is:

$$
\hat{\tau}^T(x, t) = \hat{\mu}_t(x) - \hat{\mu}_0(x)
$$

The data-splitting problem is now more severe: with 5 arms, each model has only 20% of the data. The variance of each $\hat{\mu}_t$ is roughly $5\times$ higher than if we used all the data.

#### X-learner: $K$ binary subproblems

Run $K = 4$ separate binary X-learners, each comparing one tier against control. For each tier $t$:

- Stage 1: fit $\hat{\mu}_0(x)$ on control data, $\hat{\mu}_t(x)$ on tier-$t$ data
- Stage 2: impute treatment effects and fit $\hat{\tau}_t(x)$
- Stage 3: propensity-weight

This is clean and modular — each subproblem is a standard binary X-learner. The downside: the control model $\hat{\mu}_0$ is re-estimated $K$ times (once per subproblem), and structure across tiers (e.g., the CATE at $10 should be roughly twice the CATE at $5 for a linear responder) is not shared. In practice, sharing the $\hat{\mu}_0$ model across all $K$ subproblems is a straightforward improvement.

#### Causal forests: multi-arm GRF

The `grf` package natively supports multi-arm treatment via `multi_arm_causal_forest()`. The splitting criterion generalizes to maximize heterogeneity across a *vector* of treatment contrasts simultaneously. This is the most principled multi-treatment approach — it shares information across tiers and jointly estimates the full CATE vector $\boldsymbol{\tau}(x)$ with confidence intervals.

#### Practical recommendation for Shop Cash

For a first iteration, **T-learner with gradient boosted trees** is a strong, interpretable baseline — it's simple to implement, easy to debug, and gives you a working system quickly. As a second model, **multi-arm causal forests** provide uncertainty quantification and a check on the T-learner estimates. If the two approaches agree on who the high-CATE users are, you have more confidence in the allocation. If they disagree, the causal forest confidence intervals can help diagnose where and why.

---

### 2.7 Practical modeling decisions

With the methods established, this section addresses three choices that cut across all of them and significantly affect real-world performance: what features to use, how to encode the treatment, and how to handle the outcome distribution.

**Feature engineering.** With full user history available, standard e-commerce features apply: recency (days since last purchase), frequency (orders in last $N$ days), monetary (total GMV), category affinity, app engagement (sessions, clicks, cart additions), push notification responsiveness, and prior reward redemption history. The goal is to capture dimensions along which treatment response might vary — a user who clicks frequently but rarely converts might respond very differently to a cash incentive than a user who already converts regularly.

**Encoding the treatment.** For the S-learner, how you encode $T$ matters. Three options:
- **Dummy variables**: one-hot encoding of each tier. Most flexible, but the model learns no ordering.
- **Numeric**: treat $T$ as a continuous variable (the dollar amount). Imposes a monotonic structure (higher reward → higher response), which may or may not be realistic.
- **Both**: include the dollar amount as a numeric feature *and* dummy indicators. Lets the model learn both the overall dose-response trend and tier-specific deviations.

For the T-learner and X-learner, treatment encoding is not an issue — each model is tier-specific.

**Outcome distribution.** GMV is typically right-skewed with a point mass at zero (users who don't purchase). Consider:
- Log-transforming GMV (after adding 1) to reduce skew
- Modeling the outcome in two stages: $P(\text{purchase}) \times \text{E}[\text{GMV} \mid \text{purchase}]$
- Using quantile regression or distribution-aware losses if tail behavior matters

These choices interact with the CATE estimator: a T-learner with gradient boosted trees is robust to skewed outcomes (trees handle non-normality naturally), while a causal forest's leaf-level difference-in-means estimates can be sensitive to heavy-tailed outcome distributions.

---

### Part II wrap-up

The four methods form a spectrum of complexity and assumptions:

| Method | Models fit | Key strength | Key weakness |
|--------|:---------:|:------------:|:------------:|
| S-learner | 1 | Sample-efficient, simple | Regularization bias toward $\hat{\tau} = 0$ |
| T-learner | $K+1$ | No bias against finding effects | High variance from data splitting |
| X-learner | $2(K+1)$ + propensity | Efficient with imbalanced groups | Complexity; multi-stage error propagation |
| Causal forest | Forest | Confidence intervals; adaptive | Computationally heavier; less interpretable |

For the Shop Cash project, the practical recommendation is to start with a **T-learner baseline** (simple, fast, debuggable) and follow up with **multi-arm causal forests** for uncertainty quantification. The CATE estimates from either method feed directly into the allocation optimizer from the [objective design](../uplift_modeling/objective_design.md) — the uplift model produces $\hat{\tau}(x, t)$ for each user-tier pair, and the knapsack solver assigns rewards to maximize total net uplift under the budget.

The next critical question — and the subject of the [next note](curriculum.md) — is: *how do you evaluate whether these CATE estimates are any good, given that you never observe the true individual treatment effect?*

---

## Connecting Theory to Practice

### The four user segments

Before diving into implementation, it helps to have a mental model for what the CATE estimates *mean* in terms of user behavior. The uplift modeling literature classifies users into four segments based on how they respond to treatment:

![The four uplift segments: Persuadables (positive CATE — only buy if treated), Sure Things (buy regardless), Sleeping Dogs (negative CATE — buy only if NOT treated), and Lost Causes (never buy). Source: scikit-uplift documentation.](images/uplift_user_segments.jpg)
*The four user segments in uplift modeling. Only Persuadables generate positive incremental value from treatment. Source: scikit-uplift documentation (Kane et al. 2014).*

- **Persuadables** ($\tau > 0$): Users who purchase *because of* the reward. These are the only users worth targeting — they generate true incremental value. User 6 in our examples (ROIS = 1400%) is a strong persuadable.
- **Sure Things** ($\tau \approx 0$, high baseline): Users who would purchase anyway. Any reward given to them is wasted. User 1 in our examples ($25 GMV regardless of incentive) is a textbook sure thing.
- **Lost Causes** ($\tau \approx 0$, low baseline): Users who won't purchase regardless of treatment. Also a waste of budget, but at least they don't cost more than the reward itself.
- **Sleeping Dogs** ($\tau < 0$): Users who are *less* likely to purchase when treated — the reward actually backfires. This can happen if the marketing communication feels intrusive or triggers opt-outs. Sleeping dogs are rare in cash reward programs but common in email/push marketing.

**Why this matters.** The entire value of uplift modeling is in separating persuadables from the other three segments. A traditional response model (predict who will buy) conflates persuadables with sure things — both have high $P(\text{purchase} \mid \text{treated})$, but only persuadables have high $P(\text{purchase} \mid \text{treated}) - P(\text{purchase} \mid \text{control})$. The CATE is precisely the quantity that distinguishes them.

---

### End-to-end pipeline: from experiment to allocation

Here is the concrete pipeline connecting Phase 1 experimental data to reward assignments. This is what the system looks like in practice:

```
Phase 1 Data                   Uplift Model                  Allocation
┌──────────────┐         ┌─────────────────────┐        ┌──────────────┐
│ (Xᵢ, Tᵢ, Yᵢ)│────────▶│ Estimate CATE(x, t) │───────▶│ Solve         │
│ for each user│         │ for each user-tier   │        │ knapsack     │
│ in experiment│         │ pair                 │        │ given budget │
└──────────────┘         └─────────────────────┘        └──────┬───────┘
                                                               │
                                                               ▼
                                                        ┌──────────────┐
                                                        │ Reward        │
                                                        │ assignments   │
                                                        │ {user → tier} │
                                                        └──────────────┘
```

The following code snippet implements a **multi-treatment T-learner** for the Shop Cash setting using gradient boosted trees. It meets the code bar because the gap between the math ($\hat{\mu}_t(x) - \hat{\mu}_0(x)$) and the actual implementation — data splitting, model fitting per arm, reassembly into a CATE matrix — is non-trivial to see from the formulas alone.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def fit_t_learner(df, feature_cols, treatment_col='treatment', outcome_col='gmv'):
    """
    Fit a multi-treatment T-learner.
    Returns a dict of models keyed by treatment level.
    """
    models = {}
    for t in df[treatment_col].unique():
        arm_data = df[df[treatment_col] == t]
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
        model.fit(arm_data[feature_cols], arm_data[outcome_col])
        models[t] = model
    return models

def predict_cate(models, X, control_level=0):
    """
    Predict CATE for each treatment tier vs. control.
    Returns a DataFrame: rows = users, columns = tiers, values = CATE.
    """
    mu_control = models[control_level].predict(X)
    cate = {}
    for t, model in models.items():
        if t == control_level:
            continue
        cate[t] = model.predict(X) - mu_control
    return pd.DataFrame(cate, index=X.index)

# --- Usage ---
# df_experiment has columns: user features, 'treatment' ∈ {0, 5, 10, 15, 20}, 'gmv'
# feature_cols = ['recency', 'frequency', 'monetary', 'app_sessions', ...]

models = fit_t_learner(df_experiment, feature_cols)
cate_matrix = predict_cate(models, df_new_users[feature_cols])

# cate_matrix now has one column per tier: each cell is the estimated
# incremental GMV for that user at that tier vs. control.
# This feeds directly into the knapsack allocator.
```

**Reading the output.** `cate_matrix` is a DataFrame where each row is a user and each column is a reward tier. The value in cell $(u, t)$ is $\hat{\tau}(x_u, t)$ — the estimated incremental GMV user $u$ would generate from a tier-$t$ reward. This is exactly the input the [allocation optimizer](../uplift_modeling/objective_design.md) needs to solve the knapsack.

---

## Sources and Further Reading

- **Holland, P. W.** (1986). "Statistics and Causal Inference." *Journal of the American Statistical Association*, 81(396), 945–960. — The classic paper articulating the fundamental problem of causal inference.

- **Imbens, G. W. & Rubin, D. B.** (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction.* Cambridge University Press. — The definitive textbook on the potential outcomes framework.

- **Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B.** (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning." *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165. — Introduces the X-learner and provides a systematic comparison of S-, T-, and X-learners.

- **Athey, S. & Imbens, G. W.** (2016). "Recursive partitioning for heterogeneous causal effects." *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360. — The foundational paper on causal trees, introducing the honesty concept.

- **Wager, S. & Athey, S.** (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests." *Journal of the American Statistical Association*, 113(523), 1228–1242. — Extends causal trees to forests and proves asymptotic normality (the basis for confidence intervals).

- **Athey, S., Tibshirani, J., & Wager, S.** (2019). "Generalized Random Forests." *Annals of Statistics*, 47(2), 1148–1178. — The GRF framework that unifies causal forests, including multi-treatment extensions.

- **Gutierrez, P. & Gérardy, J. Y.** (2017). "Causal Inference and Uplift Modelling: A Review of the Literature." *JMLR Workshop and Conference Proceedings*, 67, 1–13. — A practical survey of uplift modeling methods with an industry perspective.
