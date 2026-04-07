# Doubly Robust Estimation: Derivation from $V(\pi)$

## Setup and definitions

- $x$ — context (user features), drawn from some distribution
- $a \in \{1, \dots, K\}$ — actions (message variants)
- $r(x, a)$ — true expected reward for action $a$ in context $x$ (unknown to us)
- $\hat{r}(x, a) = \hat{W}[a] \cdot x$ — **reward model**, a learned linear model with one weight vector per action, trained on observed cost targets via squared loss. Can be evaluated for any action, not just the one chosen. Observed rewards enter indirectly: they shaped $\hat{W}$ during training.
- $\pi(a \mid x)$ — **target policy**, a probability distribution over actions defined by $\varepsilon$-greedy.  The action with the highest $\hat{r}$ gets most of the mass; the rest share $\varepsilon/K$ each:
$$\pi(a \mid x) = \begin{cases} 1 - \varepsilon + \varepsilon/K & \text{if } a = \arg\max_{a'} \hat{r}(x, a') \\ \varepsilon/K & \text{otherwise} \end{cases}$$

- $\mu(a \mid x)$ — **logging policy**, the policy that actually chose the actions when the data was collected
- $N$ logged observations $\{(x_i, a_i, r_i, \mu_i)\}_{i=1}^N$ collected under $\mu$:
  - $a_i$ — the action $\mu$ chose for context $x_i$
  - $r_i$ — the observed reward (click = 1, no click = 0) for that action only
  - $\mu_i = \mu(a_i \mid x_i)$ — the propensity (probability $\mu$ assigned to $a_i$)

## What we want

Estimate the expected reward of a target policy $\pi$, using data collected under a different policy $\mu$:

$$V(\pi) = \mathbb{E}_x\!\left[\sum_{a=1}^{K} \pi(a \mid x) \cdot r(x, a)\right]$$

In words: for each context, sum each action's reward weighted by how often $\pi$ would choose it, then average over contexts.

---

## Attempt 1: Direct Method — just use the model

Plug $\hat{r}$ in for the unknown $r(x, a)$ and compute the $\pi$-weighted sum over all actions:

$$\hat{V}_{\text{DM}}(\pi) = \frac{1}{N} \sum_{i=1}^{N} \sum_{a=1}^{K} \pi(a \mid x_i) \cdot \hat{r}(x_i, a)$$

Low variance (no importance weights), but biased if $\hat{r} \neq r$ — and it will be, especially for actions the policy rarely tries.  The observed rewards are only used indirectly (through $\hat{W}$), so model errors propagate with no correction.

---





## Attempt 2: IPS — ignore the model, reweight the data

Forget the model entirely.  We want the value of a **target policy** $\pi$:


$$
V(\pi) \;=\; \mathbb{E}_{x\sim p(x)}\!\left[\sum_{a\in\mathcal A} \pi(a\mid x)\, r(x,a)\right].
$$

Equivalently, as an iterated expectation:

$$
V(\pi)
=
\mathbb{E}_{x\sim p(x)}\!\left[\mathbb{E}_{a\sim \pi(\cdot\mid x)}\big[r(x,a)\big]\right].
$$

However, our data were not collected by $\pi$. They were collected by a logging policy $\mu$. Concretely, each logged interaction $i\in\{1,\dots,N\}$ is generated as:

- $x_i \sim p(x)$ (the environment/context distribution),
- $a_i \sim \mu(\cdot\mid x_i)$ (the logging policy),
- we observe a reward $r_i$ (a realized sample of $r$ for that $(x_i,a_i)$).

**Step 1:** Rewrite the expectation under $\pi$ as an expectation under $\mu$ (importance sampling). 

Fix a context $x$. Expand the inner expectation as a sum:
$$
\mathbb{E}_{a\sim \pi(\cdot\mid x)}[r(x,a)]
=
\sum_{a\in\mathcal A} \pi(a\mid x)\, r(x,a).
$$

Multiply and divide by $\mu(a\mid x)$. This is valid under the **overlap / support** assumption:
$$
\forall x,a:\quad \pi(a\mid x)>0 \implies \mu(a\mid x)>0.
$$

Then:
$$
\sum_{a} \pi(a\mid x)\, r(x,a)
=
\sum_{a} \mu(a\mid x)\, \frac{\pi(a\mid x)}{\mu(a\mid x)}\, r(x,a)
=
\mathbb{E}_{a\sim \mu(\cdot\mid x)}\!\left[\frac{\pi(a\mid x)}{\mu(a\mid x)}\, r(x,a)\right].
$$

That's the importance sampling identity: an expectation under $\pi$ rewritten as an expectation under $\mu$, with the ratio $\pi/\mu$ correcting for the difference in distributions.

Substitute back into the full expression:
$$
V(\pi)
=
\mathbb{E}_{x\sim p(x)}\!\left[
\mathbb{E}_{a\sim \mu(\cdot\mid x)}\!\left[\frac{\pi(a\mid x)}{\mu(a\mid x)}\, r(x,a)\right]
\right].
$$

**Step 3:** “Collapse” the nested expectations into a single expectation over the joint logging distribution

Define
$$
f(x,a) \;:=\; \frac{\pi(a\mid x)}{\mu(a\mid x)}\, r(x,a).
$$

Then the previous line is
$$
V(\pi) \;=\; \mathbb{E}_{x\sim p(x)}\Big[\mathbb{E}_{a\sim \mu(\cdot\mid x)}[f(x,a)]\Big].
$$

This nested expectation is exactly an expectation over the **joint** distribution of $(x,a)$ induced by the logging process:
$$(x,a) \sim p(x)\,\mu(a\mid x),$$
because
$$\mathbb{E}_{x\sim p(x)}\Big[\mathbb{E}_{a\sim \mu(\cdot\mid x)}[f(x,a)]\Big]
=
\sum_x p(x)\sum_a \mu(a\mid x) f(x,a)
=
\mathbb{E}_{(x,a)\sim p(x)\mu(a\mid x)}[f(x,a)].$$
So we can write:
$$V(\pi)
=
\mathbb{E}_{(x,a)\sim p(x)\mu(a\mid x)}\!\left[\frac{\pi(a\mid x)}{\mu(a\mid x)}\, r(x,a)\right].$$

**Step 4:** Replace the population expectation with a sample average (Monte Carlo / LLN)

The identity from Step 3 can be written in a way that matches the log directly by treating the reward as a random variable:
$$V(\pi)
=
\mathbb{E}_{(X,A,R)\sim p(x)\mu(a\mid x)p(r\mid x,a)}\!\left[\frac{\pi(A\mid X)}{\mu(A\mid X)}\, R\right].$$
Here, $\pi(a\mid x)$ and $\mu(a\mid x)$ are (typically) known probability functions you can evaluate on logged pairs $(x_i,a_i)$, while $R$ is random (we only ever see one draw $r_i$ per row). Define the per-row IPS term
$$Z_i \;:=\; \frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}\, r_i.$$
Under the usual assumption that the log entries are i.i.d. (or at least satisfy a law-of-large-numbers condition), $\{Z_i\}_{i=1}^N$ are i.i.d. samples of the same random variable
$$Z \;=\; \frac{\pi(A\mid X)}{\mu(A\mid X)}\, R,$$
so a natural Monte Carlo estimator of $\mathbb{E}[Z]$ is the sample mean:
$$\hat{V}_{\text{IPS}}(\pi)
=
\frac{1}{N}\sum_{i=1}^N Z_i
=
\frac{1}{N}\sum_{i=1}^N \frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}\, r_i.$$
**Unbiasedness:** IPS is unbiased when the propensities are correct and there is overlap. More precisely, if the logged $a_i$ are truly drawn from the same $\mu(\cdot\mid x_i)$ that appears in the denominator (and $\mu(a\mid x)>0$ wherever $\pi(a\mid x)>0$), then
$$\mathbb{E}\!\left[\hat{V}_{\text{IPS}}(\pi)\right]
=
V(\pi).$$
This is not because the summand is deterministic—$r_i$ is random—but because the importance-weighting identity makes $\mathbb{E}[Z]=V(\pi)$, and the sample mean preserves the expectation: $\mathbb{E}[\frac{1}{N}\sum_i Z_i]=\frac{1}{N}\sum_i \mathbb{E}[Z_i]=\mathbb{E}[Z]$.

**High variance:** Even though the estimator is unbiased, its variance can be large:
$$\mathrm{Var}\!\left(\hat{V}_{\text{IPS}}(\pi)\right)
=
\frac{1}{N}\,\mathrm{Var}(Z)
\quad\text{(in the i.i.d. case).}$$
The key issue is the weight
$$w_i \;=\; \frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}.$$
When $\mu(a_i\mid x_i)$ is small (i.e., the logger rarely took that action in that context) but $\pi(a_i\mid x_i)$ is not small, $w_i$ can become very large, so a single term $w_i r_i$ can dominate the sum. In extreme cases this yields heavy-tailed estimates; if the reward is unbounded and/or the importance weights have heavy tails, the variance can be enormous or even undefined. This is why practical variants often introduce techniques like clipping/capping weights, self-normalized IPS, or doubly robust estimators to trade a bit of bias for much lower variance.


**Mechanics:**

IPS estimator “reweights” each logged reward by the ratio $\pi/\mu$, correcting for the fact that actions were sampled using $\mu$, not $\pi$. Notice what changed from the Direct Method: DM sums over **all $K$ actions** using model predictions — it never looks at what actually happened.  IPS does the opposite: it only uses the **single action $a_i$ that was actually taken** and the real reward $r_i$ we observed for it.  The importance weight $w_i = \pi(a_i \mid x_i) / \mu(a_i \mid x_i)$ corrects for the frequency mismatch without touching the model at all.

 If $\pi(a_i \mid x_i) > \mu(a_i \mid x_i)$ — meaning $\pi$ would choose this action *more often* than $\mu$ did — then $w_i > 1$ and this observation's reward gets amplified.  If $\pi < \mu$, the observation is downweighted.  The effect is to **rebalance the dataset**: observations of actions that $\pi$ favors count more, observations of actions $\pi$ avoids count less.  Since the environment's reward for a given (user, action) pair doesn't depend on which policy chose it, a dataset rebalanced to $\pi$'s action frequencies gives $\pi$'s expected reward.

Here's a worked example. We set $K = 2$, $N = 1000$, uniform $\mu$:

| | Action 1 | Action 2 |
|---|---|---|
| Times chosen by $\mu$ | 500 | 500 |
| Clicks observed | 100 | 50 |

$\pi$ puts 90% on action 1, 10% on action 2.  Weights: $w = 0.9/0.5 = 1.8$ for action 1, $w = 0.1/0.5 = 0.2$ for action 2.

$$\hat{V}_{\text{IPS}} = \frac{1}{1000}\bigl[1.8 \times 100 + 0.2 \times 50\bigr] = \frac{190}{1000} = 19\%$$

The 500 action-1 observations each count as 1.8 (effectively 900); the 500 action-2 observations each count as 0.2 (effectively 100).  This matches what $\pi$ would have generated — and 19% is indeed the CTR $\pi$ would achieve.

---

## Attempt 3: The DR insight (use the model as a baseline, IPS-correct the residual)

As usual, let $\pi$ be the target policy and $\mu$ the logging policy. Logged data are generated by
$$x \sim p(x),\quad a \sim \mu(\cdot\mid x),\quad R \sim p(\cdot\mid x,a),$$

and we observe $r_i := R_i$ for each logged triple $(x_i,a_i,r_i)$. Assume overlap: for all $x,a$, $\pi(a\mid x)>0 \Rightarrow \mu(a\mid x)>0$.

Define the (unknown) conditional mean reward
$$r(x,a) \;:=\; \mathbb{E}_{R\sim p(\cdot\mid x,a)}[R],$$
and let $\hat r(x,a)$ be a fitted model intended to approximate $r(x,a)$.
Step 1: Start from the definition of the value
$$V(\pi)
\;=\;
\mathbb{E}_{x\sim p(x)}\!\left[\sum_{a\in\mathcal A}\pi(a\mid x)\, r(x,a)\right].$$

**Step 1:** Add and subtract the model (baseline + residual)
Add and subtract $\hat r(x,a)$ inside the sum:
$$V(\pi)
=
\mathbb{E}_{x\sim p(x)}\!\left[\sum_a \pi(a\mid x)\,\hat r(x,a)\right]
+
\mathbb{E}_{x\sim p(x)}\!\left[\sum_a \pi(a\mid x)\,\bigl(r(x,a)-\hat r(x,a)\bigr)\right].$$


**Term (A)** is the Direct Method part, which is computable because $\hat r(x,a)$ is available for all actions:
$$\mathbb{E}_{x\sim p(x)}\!\left[\sum_a \pi(a\mid x)\,\hat r(x,a)\right],$$



**Term (B)** is the residual under $\pi$, which is **not** directly computable from bandit feedback because $r(x,a)$ is unknown for unchosen actions:
$$\mathbb{E}_{x\sim p(x)}\!\left[\sum_a \pi(a\mid x)\,\bigl(r(x,a)-\hat r(x,a)\bigr)\right],$$


**Step 3:** Rewrite the residual term using importance sampling

Fix a context $x$. The inner sum is an expectation over $a\sim \pi(\cdot\mid x)$:
$$\sum_a \pi(a\mid x)\,\bigl(r(x,a)-\hat r(x,a)\bigr)
=
\mathbb{E}_{a\sim \pi(\cdot\mid x)}\!\left[r(x,a)-\hat r(x,a)\right].$$
Apply the importance sampling identity (using overlap):
$$\mathbb{E}_{a\sim \pi(\cdot\mid x)}[g(x,a)]
=
\mathbb{E}_{a\sim \mu(\cdot\mid x)}\!\left[\frac{\pi(a\mid x)}{\mu(a\mid x)}\,g(x,a)\right].$$
With $g(x,a)=r(x,a)-\hat r(x,a)$, we get
$$\sum_a \pi(a\mid x)\,\bigl(r(x,a)-\hat r(x,a)\bigr)
=
\mathbb{E}_{a\sim \mu(\cdot\mid x)}\!\left[\frac{\pi(a\mid x)}{\mu(a\mid x)}\,\bigl(r(x,a)-\hat r(x,a)\bigr)\right].$$
Now put back the outer expectation over contexts (explicitly):
$$\mathbb{E}_{x\sim p(x)}\!\left[\sum_a \pi(a\mid x)\,\bigl(r(x,a)-\hat r(x,a)\bigr)\right]
=
\mathbb{E}_{x\sim p(x)}\!\left[
\mathbb{E}_{a\sim \mu(\cdot\mid x)}\!\left[\frac{\pi(a\mid x)}{\mu(a\mid x)}\,\bigl(r(x,a)-\hat r(x,a)\bigr)\right]
\right].$$
Finally, connect this to what we actually observe in the log: we do not observe $r(x,a)$, but we do observe a realized reward $R$ with conditional mean $r(x,a)$. Using $r(x,a)=\mathbb{E}_{R\sim p(\cdot\mid x,a)}[R]$,
$$\mathbb{E}_{x\sim p(x)}\mathbb{E}_{a\sim \mu(\cdot\mid x)}\!\left[\frac{\pi(a\mid x)}{\mu(a\mid x)}\,\bigl(r(x,a)-\hat r(x,a)\bigr)\right]
=
\mathbb{E}_{x\sim p(x)}\mathbb{E}_{a\sim \mu(\cdot\mid x)}\mathbb{E}_{R\sim p(\cdot\mid x,a)}\!\left[\frac{\pi(a\mid x)}{\mu(a\mid x)}\,\bigl(R-\hat r(x,a)\bigr)\right].$$
This is the key “one logged action, one logged reward” step: the residual is evaluated only at the logged $(x,a)$, and $R$ is the observed random reward for that $(x,a)$.

**Step 4:** Combine terms and take a sample average (DR estimator)
From the add–subtract trick (Step 2),
$$V(\pi)
=
\mathbb{E}_{x\sim p(x)}\!\left[\sum_{a\in\mathcal A}\pi(a\mid x)\,\hat r(x,a)\right]
+
\mathbb{E}_{x\sim p(x)}\!\left[\sum_{a\in\mathcal A}\pi(a\mid x)\,\bigl(r(x,a)-\hat r(x,a)\bigr)\right].$$
Now rewrite the residual term using importance sampling (Step 3), being careful that the baseline term sums over all actions $a$, while the correction term uses the single logged action $A\sim \mu(\cdot\mid x)$:
$$\sum_{a\in\mathcal A}\pi(a\mid x)\,\bigl(r(x,a)-\hat r(x,a)\bigr)
=
\mathbb{E}_{A\sim \mu(\cdot\mid x)}\!\left[\frac{\pi(A\mid x)}{\mu(A\mid x)}\,\bigl(r(x,A)-\hat r(x,A)\bigr)\right].$$
Replace the unknown conditional mean $r(x,A)=\mathbb{E}_{R\sim p(\cdot\mid x,A)}[R]$ with the realized reward inside an expectation:
$$\mathbb{E}_{A\sim \mu(\cdot\mid x)}\!\left[\frac{\pi(A\mid x)}{\mu(A\mid x)}\,\bigl(r(x,A)-\hat r(x,A)\bigr)\right]
=
\mathbb{E}_{A\sim \mu(\cdot\mid x)}\mathbb{E}_{R\sim p(\cdot\mid x,A)}\!\left[\frac{\pi(A\mid x)}{\mu(A\mid x)}\,\bigl(R-\hat r(x,A)\bigr)\right].$$
Substituting back gives a single expectation under the logging data-generating process:

$$V(\pi)
=
\mathbb{E}_{x\sim p(x)}\mathbb{E}_{A\sim \mu(\cdot\mid x)}\mathbb{E}_{R\sim p(\cdot\mid x,A)}
\left[
\left(\sum_{a\in\mathcal A}\pi(a\mid x)\,\hat r(x,a)\right)
+
\frac{\pi(A\mid x)}{\mu(A\mid x)}\bigl(R-\hat r(x,A)\bigr)
\right].$$

Finally, with logged data $\{(x_i,a_i,r_i)\}_{i=1}^N$, estimate this expectation by the sample mean:
$$\boxed{
\hat V_{\mathrm{DR}}(\pi)
=
\frac{1}{N}\sum_{i=1}^N
\left[
\left(\sum_{a\in\mathcal A}\pi(a\mid x_i)\,\hat r(x_i,a)\right)
+
\frac{\pi(a_i\mid x_i)}{\mu(a_i\mid x_i)}\bigl(r_i-\hat r(x_i,a_i)\bigr)
\right].
}$$

---

## Why “doubly robust”?

First, a quick definition.

An estimator $\hat V_N$ is consistent for $V(\pi)$ if, as the number of logged samples $N\to\infty$, it converges to the true value:

$$\hat V_N \xrightarrow[N\to\infty]{} V(\pi)
\quad\text{(typically “in probability”).}$$
Informally: with enough data, it gets arbitrarily close to the truth.
The doubly robust (DR) estimator has the property that it is consistent (and in the idealized i.i.d. setting, asymptotically unbiased) if either one of two ingredients is correct:

1. the reward model $\hat r(x,a)$, or
2. the logging propensities $\mu(a\mid x)$ used in the importance weights (plus overlap).


**Case 1:** The model is perfect ($\hat r = r$)

If $\hat r(x,a)=r(x,a)$ for all $(x,a)$, then the residual has zero conditional mean:
$$\mathbb{E}[R-\hat r(X,A)\mid X,A]
=
\mathbb{E}[R-r(X,A)\mid X,A]
=
0.$$
So the correction term contributes nothing in expectation:
$$\mathbb{E}\!\left[\frac{\pi(A\mid X)}{\mu(A\mid X)}(R-\hat r(X,A))\right]=0,$$
and DR reduces to the Direct Method term:
$$V(\pi)=\mathbb{E}_{x\sim p(x)}\!\left[\sum_{a\in\mathcal A}\pi(a\mid x)\,\hat r(x,a)\right].$$
In finite samples the correction term won’t be exactly zero (because $R$ is noisy), but its expected value is zero and it shrinks as you average more data.

**Case 2:** The propensities are correct (the $\mu$ in the denominator matches the true logging policy)

Assume the logged action really is drawn as $A\sim \mu(\cdot\mid X)$, and the $\mu(A\mid X)$ you plug into the estimator is that true probability. Then the correction term is an importance-weighted estimate of the residual under $\pi$, regardless of model quality:
$$\mathbb{E}_{x\sim p(x)}\!\left[\sum_{a}\pi(a\mid x)\bigl(r(x,a)-\hat r(x,a)\bigr)\right]
=
\mathbb{E}_{x\sim p(x)}\mathbb{E}_{A\sim \mu(\cdot\mid x)}\!\left[\frac{\pi(A\mid x)}{\mu(A\mid x)}\bigl(r(x,A)-\hat r(x,A)\bigr)\right].$$
So even if $\hat r$ is badly misspecified, the IPS-style correction term targets exactly the amount by which the baseline term is “off,” and the two pieces add up to the true value in expectation. Averaging over $N$ samples then yields consistency by a law-of-large-numbers argument (again under the usual i.i.d./stationarity assumptions).

**When can DR fail?**

You typically get in trouble when both components are wrong at the same time:

- the reward model $\hat r(x,a)$ is misspecified and
- the propensities used in the weights are wrong (e.g., logging probabilities not recorded correctly, estimated poorly, or the logging policy changed over time without being accounted for).

In that case, the baseline is biased and the “correction” no longer targets the right residual, so the combined estimator can remain biased even with lots of data.

Separate but important: even when DR is consistent, it can still have high variance if propensities are very small—though typically less so than plain IPS because the correction is applied to a residual rather than the raw reward.

---

## From batch DR estimator to learning updates (how DR becomes a gradient step)
I’ll stick to costs (VW convention). If you prefer rewards, it’s the same with sign flips.

There are two different uses of DR: evaluation vs learning
- (a) Offline evaluation (OPE): You already have a fixed policy $\pi$. DR gives a scalar estimate of its expected cost:
$$V(\pi) := \mathbb{E}_{x\sim p(x)}\left[\sum_{a\in\mathcal A}\pi(a\mid x)\,c(x,a)\right].$$
- (b) Learning / optimization:
You want to fit a model (and usually a policy) from logged bandit data. The challenge is that each row only reveals the cost for the chosen action, not for all actions—so you can’t directly run ordinary supervised learning on true labels $c(x,a)$. DR (and IPS) are ways to create a training signal from partial feedback.

**Step 1:** Start from the batch DR identity (cost version)

Logged data are generated by:
$$x \sim p(x),\quad A\sim \mu(\cdot\mid x),\quad C\sim p(\cdot\mid x,A),$$
and you observe $(x_i,a_i,c_i)$ with propensity $p_i := \mu(a_i\mid x_i)$.

Let $\hat c(x,a)$ be a cost model. The DR estimator for a target policy $\pi$ can be written as:
$$\hat V_{\mathrm{DR}}(\pi)
=
\frac{1}{N}\sum_{i=1}^N
\left[
\left(\sum_{a\in\mathcal A}\pi(a\mid x_i)\,\hat c(x_i,a)\right)
+
\frac{\pi(a_i\mid x_i)}{p_i}\Big(c_i-\hat c(x_i,a_i)\Big)
\right].$$
This is a batch estimator: 
- input = dataset + $\pi$; 
- output = one number.

So how does this become something you can optimize with gradient descent?

**Step 2:** What learning needs: a per-round “label-like” quantity

In supervised learning, you have a clear label $y_i$ and you minimize a loss like $(f(x_i)-y_i)^2$.
In contextual bandits, for each round you only observe $c_i$ for the chosen action $a_i$. To fit a model that predicts costs for arbitrary $(x,a)$, you need to construct a pseudo-label (sometimes called a “pseudo-outcome”): something you can treat like a label for training, even though the full information is missing.
A common way to express DR as “a pseudo-cost for every action” on round $i$ is:
$$\boxed{
\hat c^{\mathrm{DR}}_i(a)
=
\hat c(x_i,a)
+
\frac{\mathbf{1}\{a=a_i\}}{p_i}\Big(c_i-\hat c(x_i,a_i)\Big)
}$$
Interpretation:


For unchosen actions $a\neq a_i$:
$$\hat c^{\mathrm{DR}}_i(a)=\hat c(x_i,a).$$
(no new information; you fall back to the model)


For the chosen action $a=a_i$:
$$\hat c^{\mathrm{DR}}_i(a_i)=\hat c(x_i,a_i) + \frac{c_i-\hat c(x_i,a_i)}{p_i}.$$
(baseline + propensity-corrected residual)


This makes the “scope” crystal clear: only the chosen action gets a correction.

**Important nuance:** if you update “one model per action,” unchosen actions get zero gradient

Suppose you parameterize $\hat c$ as separate linear models:
$$\hat c(x,a)=w_a^\top x.$$
If you literally do squared-loss regression for each action using target $\hat c^{\mathrm{DR}}_i(a)$:
$$w_a \leftarrow w_a - \eta \Big(w_a^\top x_i - \hat c^{\mathrm{DR}}_i(a)\Big)x_i,$$
then for unchosen actions $a\neq a_i$,
$$\hat c^{\mathrm{DR}}_i(a)=\hat c(x_i,a)=w_a^\top x_i,$$
so the residual is zero and the update is exactly zero.
That’s not an accident—it reflects the bandit constraint: on round $i$ you didn’t observe anything about those actions, so there’s no new per-action learning signal for them unless your parameterization shares information across actions.
So there are two standard ways forward:

- You only update the chosen action’s parameters (works but can be sample-inefficient if each action has its own independent model).
- You use shared parameters via action-dependent features (this is the most common in practice, and it’s where ADF fits in, which we cover in the next section).

**Step 4:** The actual loss and gradient calculations (how DR becomes a GD step)

Assume a shared-parameter model with action-dependent features (ADF-style parameterization):
$$\hat c(x,a) = w^\top \phi(x,a).$$
On round $i$, you observe $(x_i, a_i, c_i)$ and propensity $p_i=\mu(a_i\mid x_i)$.

**4.1:** Define the DR pseudo-label for the chosen action
First compute the model prediction on the chosen action:
$$\hat c_i := \hat c(x_i,a_i) = w^\top \phi(x_i,a_i).$$
Define the DR pseudo-label (for the chosen action only):
$$y_i^{\mathrm{DR}}
:=
\hat c(x_i,a_i) + \frac{c_i - \hat c(x_i,a_i)}{p_i}.$$

**4.2:** Define the per-round squared loss
Treat $(\phi(x_i,a_i), y_i^{\mathrm{DR}})$ as a regression example and use squared loss:
$$\ell_i(w)
=
\frac{1}{2}\Big(\hat c(x_i,a_i) - y_i^{\mathrm{DR}}\Big)^2
=
\frac{1}{2}\Big(w^\top \phi(x_i,a_i) - y_i^{\mathrm{DR}}\Big)^2.$$

**4.3:** Compute the gradient
Differentiate w.r.t. $w$:
$$\nabla_w \ell_i(w)
=
\Big(w^\top \phi(x_i,a_i) - y_i^{\mathrm{DR}}\Big)\,\phi(x_i,a_i).$$
Now simplify the residual term using the definition of $y_i^{\mathrm{DR}}$. Let $\phi_i := \phi(x_i,a_i)$ for brevity and $\hat c_i = w^\top \phi_i$:
$$\hat c_i - y_i^{\mathrm{DR}}
=
\hat c_i - \left(\hat c_i + \frac{c_i - \hat c_i}{p_i}\right)
=
-\frac{c_i - \hat c_i}{p_i}
=
\frac{\hat c_i - c_i}{p_i}.$$
So the gradient becomes:
$$\boxed{
\nabla_w \ell_i(w)
=
\frac{\hat c(x_i,a_i) - c_i}{p_i}\,\phi(x_i,a_i).
}$$
And the SGD update is:
$$\boxed{
w \leftarrow w - \eta \,\frac{\hat c(x_i,a_i) - c_i}{p_i}\,\phi(x_i,a_i).
}$$
This is exactly a standard squared-loss gradient step, but with the error scaled by $1/p_i$ (importance weighting). The DR perspective explains why this is a sensible target under bandit feedback (and how it connects back to the batch DR estimator / variance reduction logic).

4.4 Why unchosen actions give zero gradient in the “one model per action” view
If instead you had separate models $\hat c(x,a)=w_a^\top x$ and tried to regress to per-action pseudo-costs $\hat c_i^{\mathrm{DR}}(a)$ for all actions with squared loss
$$\ell_i(w_a)
= \frac{1}{2}\Big(w_a^\top x_i - \hat c_i^{\mathrm{DR}}(a)\Big)^2,$$
then for any unchosen action $a\neq a_i$, the pseudo-cost equals the current prediction:
$$\hat c_i^{\mathrm{DR}}(a)=\hat c(x_i,a)=w_a^\top x_i,$$
so the residual is $0$ and therefore
$$\nabla_{w_a}\ell_i(w_a)=0.$$
That’s expected: on round $i$ you only gain new information about the chosen action $a_i$. If you want updates to generalize across actions even when only one is observed, you need shared parameters, i.e. $\hat c(x,a)=w^\top \phi(x,a)$—which is precisely what ADF enables cleanly.


**Pseudocode (literal arithmetic):**
```python
# Logged event i: (x_i, a_i, c_i, p_i)
# Model: c_hat(x,a) = w^T phi(x,a)

# Predict chosen action cost
c_hat_ai = dot(w, phi(x_i, a_i))

# DR pseudo-label for chosen action
y_dr = c_hat_ai + (c_i - c_hat_ai) / p_i

# Squared-loss gradient step on chosen action example
# grad = (c_hat_ai - y_dr) * phi(x_i, a_i) = ((c_hat_ai - c_i)/p_i) * phi(...)
w = w - eta * (c_hat_ai - y_dr) * phi(x_i, a_i)
```

## Action-Dependent Features (ADF): how VW represents contextual bandits and performs updates
ADF is VW’s standard way to handle contextual bandits when the feature representation depends on the action. Instead of learning $K$ completely separate models $w_a$, you learn one shared parameter vector $w$ and score each action with action-specific features.
1) Scoring actions with shared parameters
Choose a feature map $\phi(x,a)\in\mathbb{R}^d$ and model the expected cost as
$$\hat c(x,a) = w^\top \phi(x,a).$$
Given a context $x_t$ and an available action set $\mathcal A(x_t)$ (often just $\{1,\dots,K\}$), the learner computes a score (predicted cost) for each action and picks an action using an exploration policy (the logging policy) $\mu$, e.g. $\epsilon$-greedy, softmax, etc.
2) What an ADF training example looks like
On round $t$, you construct an ADF example that conceptually contains:

the shared context features (can be included inside $\phi(x,a)$), and
one “line” per action $a\in\mathcal A(x_t)$ containing that action’s features.

You then reveal only:

the chosen action $a_t$,
its observed cost $c_t$,
its propensity $p_t = \mu(a_t\mid x_t)$.

Everything else is counterfactual.
This is the core bandit situation: one label per round, but many candidate actions.
3) How VW turns a bandit event into a supervised-like update
In ADF, the most common reduction is: treat the chosen action’s observation as a weighted regression/classification signal on $\phi(x_t,a_t)$, with an importance weight based on the propensity.
There are two closely related “targets” you can talk about:
3.1 IPS-style regression target (no baseline)
A very direct view is IPS: the chosen action provides an unbiased signal for its expected cost, scaled by $1/p_t$. In a squared-loss regression picture, you minimize the per-round loss
$$\ell_t(w) = \frac{1}{2}\cdot \frac{1}{p_t}\Big(w^\top \phi(x_t,a_t) - c_t\Big)^2.$$
Gradient:
$$\nabla_w \ell_t(w) = \frac{1}{p_t}\Big(w^\top \phi(x_t,a_t) - c_t\Big)\phi(x_t,a_t),$$
SGD update:
$$w \leftarrow w - \eta \cdot \frac{1}{p_t}\Big(\hat c(x_t,a_t)-c_t\Big)\phi(x_t,a_t).$$
This matches the idea: you only update on the chosen action, but because parameters are shared through $\phi(x,a)$, this update affects predictions for all actions.
3.2 DR-style update (baseline + residual)
If you also have a baseline model $\hat c(x,a)$ (possibly the same model you’re learning, or a separate one), the DR pseudo-label for the chosen action is
$$y_t^{\mathrm{DR}} = \hat c(x_t,a_t) + \frac{c_t - \hat c(x_t,a_t)}{p_t}.$$
Using squared loss on the chosen action,
$$\ell_t(w)=\frac{1}{2}\Big(w^\top \phi(x_t,a_t) - y_t^{\mathrm{DR}}\Big)^2,$$
yields gradient
$$\nabla_w \ell_t(w)
=
\Big(w^\top \phi(x_t,a_t)-y_t^{\mathrm{DR}}\Big)\phi(x_t,a_t)
=
\frac{\hat c(x_t,a_t)-c_t}{p_t}\,\phi(x_t,a_t),$$
and the same SGD update as above:
$$w \leftarrow w - \eta \cdot \frac{\hat c(x_t,a_t)-c_t}{p_t}\,\phi(x_t,a_t).$$
So in the “chosen-action regression” view, DR shows up as learning from an importance-weighted residual; the baseline primarily reduces variance and ties back to the batch DR estimator used for offline evaluation.
4) Why ADF matters (what it fixes vs. “one model per action”)
If you tried to learn independent $w_a$ with features $x$ only, you’d only update $w_{a_t}$ on round $t$. With ADF and shared $w$, the update on $\phi(x_t,a_t)$ changes parameters that are reused across other actions’ features $\phi(x_t,a)$. This is how you “learn about unchosen actions” indirectly: not by fake labels, but by generalization through shared structure.
ADF also naturally supports:

large $K$ (actions represented by features rather than separate parameter vectors),
context-dependent available actions $\mathcal A(x)$ (varying action sets),
structured actions (slates, products with attributes, etc.).

5) How ADF connects back to policy optimization
Once you have a cost predictor $\hat c(x,a)$, a common greedy policy is:
$$\pi_{\text{greedy}}(x) = \arg\min_{a\in\mathcal A(x)} \hat c(x,a).$$
During learning, exploration (logging) makes sure each action is chosen with some probability $p_t>0$, enabling unbiased / consistent learning signals.

6) Minimal “implementation mental model” for VW ADF
Per round $t$:

Build action-dependent features $\phi(x_t,a)$ for each available action $a$.
Choose $a_t$ using an exploration policy $\mu(\cdot\mid x_t)$; record $p_t=\mu(a_t\mid x_t)$.
Observe cost $c_t$.
Update using only the chosen action’s features, scaled by $1/p_t$:

$$w \leftarrow w - \eta \cdot \frac{\hat c(x_t,a_t)-c_t}{p_t}\,\phi(x_t,a_t).$$
(Depending on the exact VW reduction and loss, the update is implemented via example weights / importance weights, but mathematically it’s this idea.)