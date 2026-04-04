# MLE, MAP, and the Bayesian Approach

---

## 1. The Core Question: What Do We Mean by "Best Parameters"?

A parameterized ML model has some weight vector **w**. Given data $\mathcal{D}$, we need to do something useful with **w** — commit to a value, or maintain a distribution over it. There are three progressively richer answers to this question, each handling uncertainty about **w** differently.

| Approach | What it produces | Uncertainty about **w** |
|---|---|---|
| MLE | Point estimate — maximize likelihood | Ignored |
| MAP | Point estimate — maximize posterior | Used to regularize, then discarded |
| Full Bayesian | Distribution $p(\mathbf{w} \mid \mathcal{D})$ | Explicitly maintained throughout |

These notes build each approach carefully, connect them algebraically and conceptually, and run all three on the same concrete regression problem so the differences are tangible rather than abstract. We start with MLE and show how its failure modes motivate MAP; MAP's residual limitation then motivates the full Bayesian treatment. The Bayesian derivation culminates in a result that neither MLE nor MAP can produce: a predictive distribution whose uncertainty grows honestly with extrapolation distance.

---

### Running Example Setup

We use this throughout. Suppose we observe three data points:

| $x$ | $y$ |
|-----|-----|
| $-1$ | $0.5$ |
| $0$ | $1.2$ |
| $1$ | $1.8$ |

The model is simple linear regression with an intercept: $y = w_0 + w_1 x + \epsilon$, with noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ and $\sigma^2 = 0.1$ known. We want to estimate $\mathbf{w} = [w_0, w_1]^T$.

In matrix form, the design matrix and target vector are:

$$\boldsymbol{\Phi} = \begin{bmatrix} 1 & -1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}, \qquad \mathbf{y} = \begin{bmatrix} 0.5 \\ 1.2 \\ 1.8 \end{bmatrix}$$

---

## 2. Maximum Likelihood Estimation (MLE)

**Roadmap.** This section introduces MLE, derives its solution for linear regression, and shows that the familiar least-squares formula is not a heuristic but a direct consequence of assuming Gaussian noise. We close by identifying MLE's structural failure mode — overfitting in small-data settings — that motivates MAP.

### 2.1 The Principle

MLE asks: *what value of **w** makes the data I observed most probable?*

$$\hat{\mathbf{w}}_{\text{MLE}} = \arg\max_{\mathbf{w}} \; p(\mathcal{D} \mid \mathbf{w})$$

This is the **frequentist** stance: parameters are fixed (if unknown) constants, and data are the random things. The parameter that best "explains" the observed data wins.

### 2.2 Linear Regression: MLE = OLS

This connection is usually stated without proof, but it falls out cleanly. Assuming observations are conditionally independent given **w**, and $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$:

$$p(\mathcal{D} \mid \mathbf{w}) = \prod_{i=1}^N p(y_i \mid x_i, \mathbf{w}) = \prod_{i=1}^N \mathcal{N}(y_i;\, \mathbf{w}^T \boldsymbol{\phi}_i,\, \sigma^2)$$

where $\boldsymbol{\phi}_i = [1, x_i]^T$ is the feature vector for input $x_i$. Taking the log (which preserves the argmax):

$$\log p(\mathcal{D} \mid \mathbf{w}) = \sum_{i=1}^N \log \mathcal{N}(y_i;\, \mathbf{w}^T \boldsymbol{\phi}_i,\, \sigma^2) = \sum_{i=1}^N \left[ -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - \mathbf{w}^T \boldsymbol{\phi}_i)^2}{2\sigma^2} \right]$$

The first term is a constant in **w**. Maximizing over **w** is therefore equivalent to minimizing the sum of squared residuals:

$$\hat{\mathbf{w}}_{\text{MLE}} = \arg\min_{\mathbf{w}} \sum_{i=1}^N (y_i - \mathbf{w}^T \boldsymbol{\phi}_i)^2 = \arg\min_{\mathbf{w}} \|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2$$

**This is exactly ordinary least squares.** OLS is not a heuristic — it is MLE under a Gaussian noise model. The noise assumption implies the objective. Setting the gradient to zero:

$$\nabla_{\mathbf{w}} \|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2 = -2\boldsymbol{\Phi}^T(\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}) = 0 \implies \boldsymbol{\Phi}^T\boldsymbol{\Phi}\,\hat{\mathbf{w}} = \boldsymbol{\Phi}^T \mathbf{y}$$

$$\boxed{\hat{\mathbf{w}}_{\text{MLE}} = (\boldsymbol{\Phi}^T \boldsymbol{\Phi})^{-1} \boldsymbol{\Phi}^T \mathbf{y}}$$

### 2.3 Running Example: MLE Solution

For our 3-point dataset:

$$\boldsymbol{\Phi}^T\boldsymbol{\Phi} = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}, \qquad \boldsymbol{\Phi}^T\mathbf{y} = \begin{bmatrix} 3.5 \\ 1.3 \end{bmatrix}$$

Since $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ is diagonal — the off-diagonal entry is $\sum_i x_i = -1 + 0 + 1 = 0$, so the intercept and slope columns of $\boldsymbol{\Phi}$ are orthogonal, a consequence of the $x$ values being symmetric around zero:

$$\hat{\mathbf{w}}_{\text{MLE}} = \begin{bmatrix} 3.5/3 \\ 1.3/2 \end{bmatrix} = \begin{bmatrix} 1.167 \\ 0.650 \end{bmatrix}$$

The fitted line is $\hat{y} = 1.167 + 0.650\, x$. Prediction at $x_* = 2$: $\hat{y} = 2.467$.

### 2.4 The Failure Mode: MLE Overfits in Small-Sample Settings

MLE's only signal is the data in front of it. It has no mechanism to flag when an estimate looks extreme. The failure mode is clearest when the model has enough parameters to fit the noise exactly.

The snippet below demonstrates: fitting polynomials of increasing degree to just 3 points. Degree 2 achieves perfect residuals — but the resulting curve wildly misrepresents any plausible true function.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([-1, 0, 1], dtype=float)
t = np.array([0.5, 1.2, 1.8])
x_grid = np.linspace(-1.5, 1.5, 200)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, deg in zip(axes, [1, 2, 3]):
    coeffs = np.polyfit(x, t, deg)       # MLE under Gaussian noise = least squares
    ax.scatter(x, t, zorder=5, color='black', label='data')
    ax.plot(x_grid, np.polyval(coeffs, x_grid), color='steelblue')
    residuals = t - np.polyval(coeffs, x)
    ax.set_title(f'Degree {deg}  |  RSS = {(residuals**2).sum():.4f}')
    ax.set_ylim(-1, 4)

plt.suptitle('MLE (OLS) for polynomials of increasing degree on N=3 points', y=1.02)
plt.tight_layout()
plt.savefig('mle_overfit.png', dpi=120, bbox_inches='tight')
```

[FIG:ORIGINAL — polynomial MLE overfitting: three side-by-side panels showing degree-1, degree-2, and degree-3 polynomial fits to 3 data points, where degree-2 achieves perfect interpolation (RSS=0) and degree-3 has a free parameter that can be anything. Illustrates why MLE without a prior has no mechanism to prefer simpler fits.]

**The failure is structural:** MLE treats the particular sample it saw as the true distribution. With large $N$, sample statistics converge to population statistics and this is fine. With small $N$ or many parameters relative to data, MLE overfits to noise.

**Section wrap-up.** MLE gives us ordinary least squares and is optimal when data is plentiful. Its derivation (Gaussian noise → minimize squared residuals) reveals that the loss function is not arbitrary — it encodes a distributional assumption. But MLE has no brakes: given enough parameters, it will fit the noise perfectly. The fix is to introduce a prior over **w** that penalizes implausible estimates — which is exactly what MAP does.

---

## 3. Maximum A Posteriori (MAP) Estimation

**Roadmap.** MAP extends MLE by incorporating a prior belief about **w**. We derive the posterior objective, show that the Gaussian prior recovers ridge regression and the Laplace prior recovers LASSO, and work through the running example numerically. The section closes by identifying MAP's own limitation — it is still a point estimate — which motivates the full Bayesian treatment.

### 3.1 The Principle

MAP asks: *what value of **w** is most probable, given both what I believe a priori and what the data shows?*

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_{\mathbf{w}} \; p(\mathbf{w} \mid \mathcal{D})$$

By Bayes' rule, $p(\mathbf{w} \mid \mathcal{D}) \propto p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})$, so taking logs:

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max_{\mathbf{w}} \left[\log p(\mathcal{D} \mid \mathbf{w}) + \log p(\mathbf{w})\right]$$

MAP is MLE with one additional term. The prior $p(\mathbf{w})$ penalizes parameter values that are implausible before seeing data, pulling estimates away from extremes.

There is a clean frequentist interpretation of this move. MLE is an *unbiased* estimator — on average over repeated samples, it hits the true parameter — but in small-sample settings it has high *variance*: any particular sample may produce a wildly different estimate. The prior introduces *bias* (it systematically pulls estimates toward zero, even if the true parameter is nonzero) in exchange for reduced variance. When the variance reduction outweighs the added bias, the MAP estimate has lower mean squared error than MLE. This is the **bias-variance tradeoff** viewed through the lens of estimation, and it explains why regularization helps most when $N$ is small relative to the number of parameters.

### 3.2 Prior Choice = Regularizer Choice

The connection between priors and regularizers is one of the most elegant results in this area. Every standard regularization scheme turns out to be a prior in disguise.

**Gaussian prior → Ridge regression.** Place $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$. The log-prior is:

$$\log p(\mathbf{w}) = -\frac{1}{2\tau^2}\|\mathbf{w}\|^2 + \text{const}$$

Substituting the log-likelihood from Section 2.2 and dropping constants in **w**:

$$\hat{\mathbf{w}}_{\text{MAP}} = \arg\min_{\mathbf{w}} \; \frac{1}{2\sigma^2}\|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2 + \frac{1}{2\tau^2}\|\mathbf{w}\|^2$$

Multiplying through by $2\sigma^2$ (preserves argmin):

$$= \arg\min_{\mathbf{w}} \; \|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2 + \underbrace{\frac{\sigma^2}{\tau^2}}_{\lambda}\|\mathbf{w}\|^2$$

This is **ridge regression** with $\lambda = \sigma^2/\tau^2$. The regularization strength is the ratio of noise variance to prior variance: a tighter prior (small $\tau^2$) means you trust the prior more than the data, producing heavier shrinkage. Setting the gradient to zero:

$$\hat{\mathbf{w}}_{\text{MAP}} = (\boldsymbol{\Phi}^T\boldsymbol{\Phi} + \lambda I)^{-1}\boldsymbol{\Phi}^T\mathbf{y}$$

The $\lambda I$ term added to $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ guarantees invertibility even when $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ is rank-deficient — a key practical advantage of ridge over plain OLS.

**Laplace prior → LASSO.** If instead $p(w_j) \propto \exp(-|w_j|/b)$, the log-prior contributes $-\frac{1}{b}\|\mathbf{w}\|_1$, and MAP becomes LASSO. The $\ell_1$ penalty induces sparsity because the Laplace distribution has a sharp peak at zero with heavy tails — it strongly favors exactly-zero weights, unlike the Gaussian prior which merely discourages large weights.

This geometric difference is worth seeing directly:

```python
import numpy as np
import matplotlib.pyplot as plt

w = np.linspace(-4, 4, 400)
tau, b = 1.0, 1.0

gaussian = np.exp(-w**2 / (2*tau**2))
gaussian /= gaussian.max()

laplace = np.exp(-np.abs(w) / b)
laplace /= laplace.max()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(w, gaussian, label=r'Gaussian prior ($\ell_2$ / ridge)', color='steelblue', lw=2)
ax.plot(w, laplace,  label=r'Laplace prior ($\ell_1$ / LASSO)',  color='darkorange', lw=2)
ax.axvline(0, color='gray', lw=0.8, ls='--')
ax.set_xlabel('$w_j$')
ax.set_ylabel('(normalized) prior density')
ax.set_title('Gaussian vs. Laplace prior — why Laplace induces sparsity')
ax.legend()
plt.tight_layout()
plt.savefig('prior_shapes.png', dpi=120, bbox_inches='tight')
```

[FIG:ORIGINAL — side-by-side density plot of Gaussian prior vs Laplace prior over a single weight $w_j$, showing the Laplace prior's sharp cusp at zero and heavier tails compared to the smooth Gaussian. Illustrates geometrically why MAP with a Laplace prior drives weights to exactly zero (sparsity) while MAP with a Gaussian prior only shrinks them.]

**Summary table:**

| Regularizer | Equivalent prior | Effect |
|---|---|---|
| $\ell_2$ / ridge | Gaussian: $\mathcal{N}(0, \tau^2)$ | Shrinks all weights toward zero |
| $\ell_1$ / LASSO | Laplace: $\propto e^{-|w|/b}$ | Drives some weights to exactly zero (sparsity) |
| Elastic net | Product of Gaussian and Laplace | Shrinkage + sparsity |

**Takeaway:** Choosing a regularizer is equivalent to choosing a prior. The "regularization strength" $\lambda$ encodes the ratio of noise variance to prior variance. This is not just a pretty connection — it gives regularization a principled interpretation and offers a route to setting $\lambda$ via prior beliefs rather than cross-validation alone.

### 3.3 Running Example: MAP Solution

Using $\sigma^2 = 0.1$ and $\tau^2 = 1$ gives $\lambda = \sigma^2/\tau^2 = 0.1$:

$$\boldsymbol{\Phi}^T\boldsymbol{\Phi} + \lambda I = \begin{bmatrix} 3.1 & 0 \\ 0 & 2.1 \end{bmatrix}$$

$$\hat{\mathbf{w}}_{\text{MAP}} = \begin{bmatrix} 3.5/3.1 \\ 1.3/2.1 \end{bmatrix} = \begin{bmatrix} 1.129 \\ 0.619 \end{bmatrix}$$

Compare to MLE: $[1.167,\; 0.650]$. Both weights have been **shrunk toward zero**. The shrinkage is modest here because $\lambda = 0.1$ is small relative to the diagonal entries of $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$. With fewer data points or a tighter prior (smaller $\tau^2$, larger $\lambda$), the prior would dominate more.

Prediction at $x_* = 2$: $\hat{y} = 1.129 + 0.619 \times 2 = 2.367$ (vs. MLE's $2.467$ — the slope was shrunk).

### 3.4 The Limitation of MAP

MAP is still a **point estimate**. It picks the mode of the posterior and discards everything else about its shape. With only 3 data points, the posterior over **w** is broad — many weight settings are nearly as plausible as the MAP solution — but MAP gives no way to express this. Predictions carry no uncertainty about **w**: we predict as if **w** is known exactly once we have the MAP estimate.

**Section wrap-up.** MAP = MLE + log-prior, and every standard regularizer is a prior in disguise. This gives regularization a probabilistic interpretation: you are encoding a belief about the scale and sparsity of the weights before seeing data. But MAP still produces a single point estimate — the mode of the posterior — and throws away everything else. The posterior is a distribution, not a point, and discarding it loses information that becomes important in low-data settings. The Bayesian approach keeps it.

---

## 4. The Bayesian Approach: Maintaining the Full Posterior

**Roadmap.** This section develops the Bayesian framework in full generality before we specialize to linear regression. We introduce the posterior and marginal likelihood, define conjugate priors and explain why they matter computationally, derive the predictive distribution and explain the two sources of uncertainty it captures, flag a subtle but important point about marginal independence, and close with the general Bayesian regression formulation showing all conditioning explicitly. The concrete closed-form derivation follows in Section 5.

### 4.1 The Posterior Distribution

Rather than collapsing to a point estimate, the Bayesian approach maintains the complete distribution over **w** given the data:

$$p(\mathbf{w} \mid \mathcal{D}) = \frac{\overbrace{p(\mathcal{D} \mid \mathbf{w})}^{\text{likelihood}} \cdot \overbrace{p(\mathbf{w})}^{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{marginal likelihood}}}$$

The **marginal likelihood** (also called **model evidence**) is:

$$p(\mathcal{D}) = \int p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})\, d\mathbf{w}$$

It normalizes the posterior so it integrates to one. For parameter inference, it is a constant in **w** and can be ignored. But it plays a separate important role in **model comparison**: two models with different priors or structures can be ranked by their marginal likelihood — the one that assigns higher probability to the observed data (after averaging over all parameter values) is preferred. A high $p(\mathcal{D})$ means the model's prior was well-matched to the data. In principle this is the Bayesian answer to hyperparameter selection; in practice the integral is usually intractable.

### 4.2 Conjugate Priors

For most models, the posterior $p(\mathbf{w} \mid \mathcal{D})$ has no closed form because computing $p(\mathcal{D})$ requires an intractable integral. However, for certain prior-likelihood pairs, the posterior falls in the same parametric family as the prior. These are called **conjugate priors**.

**Definition:** A prior $p(\mathbf{w})$ is *conjugate* to a likelihood $p(\mathcal{D} \mid \mathbf{w})$ if the posterior $p(\mathbf{w} \mid \mathcal{D})$ is in the same parametric family as the prior.

**The key conjugate pair for regression:** A Gaussian prior on **w** is conjugate to a Gaussian likelihood (Gaussian noise model). The posterior is also Gaussian, with a closed-form mean and covariance derivable by completing the square — which is exactly what Section 5 does.

**A concrete illustration.** Suppose we have a single weight $w$ with prior $w \sim \mathcal{N}(0, \alpha^{-1})$ and observe one data point $y = w + \epsilon$, $\epsilon \sim \mathcal{N}(0, \beta^{-1})$. The posterior mean is:

$$\mathbb{E}[w \mid y] = \frac{\beta}{\alpha + \beta} y$$

This is a weighted average between the prior mean (0) and the observation ($y$), with weights proportional to their precisions. When the prior is strong ($\alpha \gg \beta$, high prior precision), the posterior mean is pulled toward 0. When the data is informative ($\beta \gg \alpha$), the posterior mean approaches $y$. The generalization to $N$ observations is immediate: $N$ i.i.d. observations each contribute precision $\beta$, so the effective data precision is $N\beta$ and the posterior mean becomes $\frac{N\beta}{\alpha + N\beta}\bar{y}$. As $N \to \infty$, this approaches $\bar{y}$ regardless of the prior — data overwhelms any finite prior belief.

**Beyond Gaussian-Gaussian:** Other standard conjugate pairs include Beta-Binomial (coin flipping), Dirichlet-Categorical (multinomial data), and Gamma-Poisson (count data). Conjugacy is the exception in modern ML — for neural networks and other nonlinear models, the posterior is intractable and must be approximated via the Laplace approximation, variational inference, or MCMC.

### 4.3 The Predictive Distribution

The central quantity we care about is not the posterior over **w** itself, but the distribution over predictions at a new test point $\mathbf{x}_*$. The Bayesian predictive distribution averages over all possible **w**, weighted by their posterior probability:

$$\boxed{p(y_* \mid \mathbf{x}_*, \mathcal{D}) = \int p(y_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w}}$$

**How to read this:** For each possible setting of **w**, $p(y_* \mid \mathbf{x}_*, \mathbf{w})$ is the prediction that **w** would make. $p(\mathbf{w} \mid \mathcal{D})$ is the posterior weight — how plausible that **w** is given all observed data. The integral takes the probability-weighted average across all settings. No single **w** dominates unless the posterior is tightly concentrated.

This simultaneously captures two sources of uncertainty:

- **Observation noise** — the irreducible randomness in the data-generating process ($\sigma^2$), present even if **w** were known exactly.
- **Parameter uncertainty** — we don't know the true **w**; the posterior spread over **w** translates into additional spread in predictions.

MLE and MAP predictions contain only observation noise. The Bayesian prediction contains both. In data-rich settings, the posterior concentrates and parameter uncertainty vanishes — all three approaches converge. In data-scarce settings, parameter uncertainty can dominate, and ignoring it produces overconfident predictions.

### 4.4 A Critical Subtlety: Marginalizing Over w Destroys Marginal Independence

It is tempting to assume that since observations are i.i.d. given **w**, the marginal likelihood should factor as $\prod_i p(y_i \mid x_i)$. This is wrong:

$$p(\mathcal{D}) = \int \left[\prod_{i=1}^N p(y_i \mid x_i, \mathbf{w})\right] p(\mathbf{w})\, d\mathbf{w} \neq \prod_{i=1}^N p(y_i \mid x_i)$$

The product over $i$ is *inside* the integral over **w**. You cannot factor the integral of a product as a product of integrals unless the factors were independent of **w** — which they are not. Writing $\prod_i p(y_i \mid x_i)$ would require a separate independent $\mathbf{w}_i$ for each observation. In our model, all observations share the same **w**, so observing $(x_1, y_1)$ gives information about **w**, which in turn shifts predictions about $y_2$.

**Intuition:** The observations are *conditionally* independent given **w**, but *marginally* dependent because they share an unknown parameter. Classic example: flip a coin with unknown bias 10 times and see 9 heads. Your prediction for flip 11 changes — the flips are not causally connected, but they all carry information about the same latent quantity $p$.

### 4.5 The General Formulation: Explicit Conditioning on Inputs

Before we specialize to the Gaussian case, it is worth writing the general Bayesian regression setup with all conditioning made explicit. For a labeled regression dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N = \{\mathbf{x}, \mathbf{y}\}$:

**Posterior over weights:**

$$p(\mathbf{w} \mid \mathbf{x}, \mathbf{y}) = \frac{p(\mathbf{y} \mid \mathbf{x}, \mathbf{w})\, p(\mathbf{w})}{p(\mathbf{y} \mid \mathbf{x})} \qquad \text{with} \qquad p(\mathbf{y} \mid \mathbf{x}) = \int p(\mathbf{y} \mid \mathbf{x}, \mathbf{w})\, p(\mathbf{w})\, d\mathbf{w}$$

Note the conditioning on **x** throughout — inputs are treated as fixed and observed; we model the conditional distribution of targets given inputs, not the joint distribution over inputs and targets.

**Predictive distribution for a new test point $(\mathbf{x}_*, y_*)$:**

$$p(y_* \mid \mathbf{x}_*, \mathbf{x}, \mathbf{y}) = \int p(y_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathbf{x}, \mathbf{y})\, d\mathbf{w}$$

The two simplifications in the integrand are worth justifying explicitly. Starting from:

$$p(y_* \mid \mathbf{x}_*, \mathbf{x}, \mathbf{y}) = \int p(y_* \mid \mathbf{x}_*, \mathbf{x}, \mathbf{y}, \mathbf{w})\, p(\mathbf{w} \mid \mathbf{x}_*, \mathbf{x}, \mathbf{y})\, d\mathbf{w}$$

1. $p(y_* \mid \mathbf{x}_*, \mathbf{x}, \mathbf{y}, \mathbf{w}) = p(y_* \mid \mathbf{x}_*, \mathbf{w})$: given **w**, the prediction for $y_*$ depends only on $\mathbf{x}_*$ — training data carry no additional information once **w** is known.
2. $p(\mathbf{w} \mid \mathbf{x}_*, \mathbf{x}, \mathbf{y}) = p(\mathbf{w} \mid \mathbf{x}, \mathbf{y})$: the test input $\mathbf{x}_*$ alone (without its label $y_*$) carries no information about **w**.

**Section wrap-up.** The Bayesian framework replaces a point estimate with a full posterior distribution over **w**, propagates that uncertainty through to predictions, and produces honest uncertainty estimates that depend on how much information the data provides at each test location. The marginal likelihood provides a principled basis for model comparison. The key computational challenge — intractability of the posterior integral — is solvable in closed form only for conjugate pairs. Gaussian-Gaussian is the main tractable case, which we now derive explicitly.

---

## 5. Bayesian Linear Regression: Full Derivation

**Roadmap.** This section derives the closed-form posterior and predictive distribution for the running example. The derivation has three parts: (1) completing the square to identify the Gaussian posterior, (2) reading off the posterior mean and covariance and connecting them back to MAP, and (3) marginalizing the posterior to get the predictive distribution, including the closed-form result for the integral of a product of two Gaussians. The key payoff is a predictive variance that is smallest near training data and grows with extrapolation distance.

### 5.1 Setup

Model: $y_i = \mathbf{w}^T \boldsymbol{\phi}_i + \epsilon_i$, with $\epsilon_i \sim \mathcal{N}(0, \beta^{-1})$. We use precision notation ($\beta = 1/\sigma^2$, $\alpha = 1/\tau^2$) because it keeps the algebra cleaner.

Prior: $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1}I)$, i.e., each weight independently has prior precision $\alpha$ (prior variance $1/\alpha$).

For our running example: $\beta = 1/\sigma^2 = 10$, $\alpha = 1$ (prior variance $= 1$).

### 5.2 Posterior: Completing the Square

The posterior is proportional to likelihood times prior:

$$p(\mathbf{w} \mid \mathcal{D}) \propto \exp\!\left(-\frac{\beta}{2}\|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2\right) \exp\!\left(-\frac{\alpha}{2}\|\mathbf{w}\|^2\right) \propto \exp\!\left(-\frac{1}{2}\left[\beta\|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2 + \alpha\|\mathbf{w}\|^2\right]\right)$$

Expanding the quadratic in the exponent (using $\|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2 = \mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\boldsymbol{\Phi}\mathbf{w} + \mathbf{w}^T\boldsymbol{\Phi}^T\boldsymbol{\Phi}\mathbf{w}$) and collecting terms by their degree in **w**:

$$= \exp\!\left(-\frac{1}{2}\left[\mathbf{w}^T\underbrace{(\alpha I + \beta\boldsymbol{\Phi}^T\boldsymbol{\Phi})}_{\equiv\, \mathbf{S}_N^{-1}}\mathbf{w} - 2\mathbf{w}^T \beta\boldsymbol{\Phi}^T\mathbf{y} + \underbrace{\beta\mathbf{y}^T\mathbf{y}}_{\text{const in }\mathbf{w}}\right]\right)$$

This is a quadratic form in **w** — the signature of a Gaussian density. We complete the square using the identity:

$$\mathbf{w}^T A \mathbf{w} - 2\mathbf{w}^T \mathbf{b} = (\mathbf{w} - A^{-1}\mathbf{b})^T A\, (\mathbf{w} - A^{-1}\mathbf{b}) - \mathbf{b}^T A^{-1}\mathbf{b}$$

Here $A = \mathbf{S}_N^{-1}$ and $\mathbf{b} = \beta\boldsymbol{\Phi}^T\mathbf{y}$, so $A^{-1}\mathbf{b} = \mathbf{S}_N (\beta\boldsymbol{\Phi}^T\mathbf{y})$. We define this as the posterior mean: $\mathbf{m}_N \equiv \mathbf{S}_N\,\beta\boldsymbol{\Phi}^T\mathbf{y}$. The term $\mathbf{b}^T A^{-1} \mathbf{b}$ does not depend on **w** and folds into the normalizing constant. The result is:

$$p(\mathbf{w} \mid \mathcal{D}) \propto \exp\!\left(-\frac{1}{2}(\mathbf{w} - \mathbf{m}_N)^T \mathbf{S}_N^{-1} (\mathbf{w} - \mathbf{m}_N)\right)$$

This is exactly $\mathcal{N}(\mathbf{w};\, \mathbf{m}_N,\, \mathbf{S}_N)$. The posterior parameters are:

$$\boxed{\mathbf{S}_N^{-1} = \alpha I + \beta \boldsymbol{\Phi}^T\boldsymbol{\Phi}, \qquad \mathbf{m}_N = \beta\, \mathbf{S}_N\, \boldsymbol{\Phi}^T\mathbf{y}}$$

**Two key observations:**

**1. The posterior mean equals the MAP estimate.** Recall from Section 3 that MAP under a Gaussian prior gives $\hat{\mathbf{w}}_{\text{MAP}} = (\boldsymbol{\Phi}^T\boldsymbol{\Phi} + \lambda I)^{-1}\boldsymbol{\Phi}^T\mathbf{y}$ with $\lambda = \sigma^2/\tau^2 = \alpha/\beta$. Substituting $\mathbf{S}_N^{-1} = \alpha I + \beta\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ and $\mathbf{m}_N = \beta\mathbf{S}_N\boldsymbol{\Phi}^T\mathbf{y}$, these are identical up to rescaling by $\beta$. For a Gaussian distribution, the mode and mean coincide — so **MAP is Bayesian inference that keeps only the mean and discards the covariance $\mathbf{S}_N$**.

**2. The posterior covariance $\mathbf{S}_N$ encodes parameter uncertainty.** Its inverse, $\mathbf{S}_N^{-1} = \alpha I + \beta\boldsymbol{\Phi}^T\boldsymbol{\Phi}$, has two additive contributions: $\alpha I$ from the prior and $\beta\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ from the data. Each new observation adds to $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$, increasing $\mathbf{S}_N^{-1}$ and shrinking $\mathbf{S}_N$. As $N \to \infty$, the posterior concentrates tightly around the true **w** and parameter uncertainty vanishes.

### 5.3 Running Example: Posterior

With $\alpha = 1$, $\beta = 10$:

$$\mathbf{S}_N^{-1} = I + 10 \cdot \begin{bmatrix}3 & 0 \\ 0 & 2\end{bmatrix} = \begin{bmatrix}31 & 0 \\ 0 & 21\end{bmatrix} \implies \mathbf{S}_N = \begin{bmatrix}1/31 & 0 \\ 0 & 1/21\end{bmatrix}$$

$$\mathbf{m}_N = 10 \cdot \mathbf{S}_N \cdot \begin{bmatrix}3.5 \\ 1.3\end{bmatrix} = \begin{bmatrix}35/31 \\ 13/21\end{bmatrix} = \begin{bmatrix}1.129 \\ 0.619\end{bmatrix}$$

As expected, $\mathbf{m}_N = \hat{\mathbf{w}}_{\text{MAP}}$. But now we also have posterior variances: $\text{Var}(w_0) = 1/31 \approx 0.032$ and $\text{Var}(w_1) = 1/21 \approx 0.048$. The slope is more uncertain than the intercept, which makes intuitive sense: slope is estimated from variation across inputs, and we only have 3 points.

**Beyond the reading: sequential (online) Bayesian updating.** One of the most elegant consequences of conjugacy is that Bayesian inference is naturally *online*. After observing a first batch of data, the posterior $p(\mathbf{w} \mid \mathcal{D}_1) = \mathcal{N}(\mathbf{m}_1, \mathbf{S}_1)$ can serve as the prior for a second batch $\mathcal{D}_2$. Because the Gaussian family is closed under this update, applying Bayes' rule again produces $\mathcal{N}(\mathbf{m}_2, \mathbf{S}_2)$ — exactly the same posterior we would obtain from processing $\mathcal{D}_1 \cup \mathcal{D}_2$ jointly. In our running example, we could process the three data points one at a time: starting from the prior $\mathcal{N}(\mathbf{0}, I)$, each observation would tighten $\mathbf{S}_N$ and shift $\mathbf{m}_N$, converging to the same $\mathbf{m}_3 = [1.129,\; 0.619]^T$ we computed above. This property — sometimes called *Bayesian updating* — means that streaming data does not require reprocessing the full dataset; the sufficient statistics ($\mathbf{S}_N^{-1}$ and $\boldsymbol{\Phi}^T \mathbf{y}$) accumulate additively.

### 5.4 Predictive Distribution: Marginalizing Out w

For a new input $\mathbf{x}_*$, define $\boldsymbol{\phi}_* = [1, x_*]^T$. The predictive distribution is:

$$p(y_* \mid \mathbf{x}_*, \mathcal{D}) = \int p(y_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w} = \int \mathcal{N}(y_*;\, \mathbf{w}^T\boldsymbol{\phi}_*,\, \beta^{-1})\, \mathcal{N}(\mathbf{w};\, \mathbf{m}_N,\, \mathbf{S}_N)\, d\mathbf{w}$$

To evaluate this, we use the **linear Gaussian marginalisation identity**: if $\mathbf{w} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Lambda}^{-1})$ and $y \mid \mathbf{w} \sim \mathcal{N}(\mathbf{A}\mathbf{w} + \mathbf{b},\, \mathbf{L}^{-1})$, then the marginal over $\mathbf{w}$ is:

$$p(y) = \int p(y \mid \mathbf{w})\, p(\mathbf{w})\, d\mathbf{w} = \mathcal{N}(y;\; \mathbf{A}\boldsymbol{\mu} + \mathbf{b},\; \mathbf{L}^{-1} + \mathbf{A}\boldsymbol{\Lambda}^{-1}\mathbf{A}^T)$$

**Intuition for this identity:** the mean of $y$ is the prediction using the prior mean of **w** ($\mathbf{A}\boldsymbol{\mu} + \mathbf{b}$), and the variance of $y$ is the observation noise ($\mathbf{L}^{-1}$) plus the variance introduced by uncertainty in **w** ($\mathbf{A}\boldsymbol{\Lambda}^{-1}\mathbf{A}^T$). The observation noise and parameter uncertainty add independently because they enter the model additively. A full derivation involves completing the square in the joint $p(y, \mathbf{w})$ and reading off the marginal — see Bishop §2.3.3 for the complete derivation.

In our case: $\mathbf{A} = \boldsymbol{\phi}_*^T$ (a row vector), $\mathbf{b} = 0$, $\boldsymbol{\mu} = \mathbf{m}_N$, $\boldsymbol{\Lambda}^{-1} = \mathbf{S}_N$, $\mathbf{L}^{-1} = \beta^{-1}$. Substituting:

$$\boxed{p(y_* \mid \mathbf{x}_*, \mathcal{D}) = \mathcal{N}\!\left(y_*;\; \underbrace{\mathbf{m}_N^T\boldsymbol{\phi}_*}_{\text{posterior mean prediction}},\; \underbrace{\beta^{-1}}_{\text{observation noise}} + \underbrace{\boldsymbol{\phi}_*^T \mathbf{S}_N \boldsymbol{\phi}_*}_{\text{parameter uncertainty}}\right)}$$

The predictive variance decomposes into two interpretable terms:

$$\sigma^2_{\text{pred}}(\mathbf{x}_*) = \underbrace{\beta^{-1}}_{\substack{\text{irreducible noise} \\ \text{present even if } \mathbf{w} \text{ were known}}} + \underbrace{\boldsymbol{\phi}_*^T \mathbf{S}_N \boldsymbol{\phi}_*}_{\substack{\text{how much } \boldsymbol{\phi}_* \text{ amplifies} \\ \text{our uncertainty about } \mathbf{w}}}$$

The second term depends on $\mathbf{x}_*$ in a meaningful way: it is small when $\boldsymbol{\phi}_*$ points in directions where $\mathbf{S}_N$ is small (directions well-constrained by data) and large when $\boldsymbol{\phi}_*$ points in poorly-constrained directions. Near the training data, we have a good estimate of how the line behaves. Far from it, the slope uncertainty gets multiplied by a large $x_*$ and the prediction becomes much less certain.

### 5.5 Running Example: Predictive Variance Grows with Extrapolation

For our example, $\mathbf{S}_N = \text{diag}(1/31,\, 1/21)$ and $\beta^{-1} = 0.1$:

$$\boldsymbol{\phi}_*^T \mathbf{S}_N \boldsymbol{\phi}_* = \frac{1}{31} + \frac{x_*^2}{21}$$

| Location | $x_*$ | Pred. mean | Noise var. | Param. uncert. | **Total var.** |
|---|---|---|---|---|---|
| Center of data | $0$ | $1.129$ | $0.100$ | $0.032$ | **$0.132$** |
| Edge of training data | $1$ | $1.748$ | $0.100$ | $0.080$ | **$0.180$** |
| Mild extrapolation | $2$ | $2.367$ | $0.100$ | $0.223$ | **$0.323$** |
| Aggressive extrapolation | $4$ | $3.605$ | $0.100$ | $0.793$ | **$0.893$** |

At $x_* = 4$, total predictive variance is nearly $7\times$ what it is at the training center. Neither MLE nor MAP would show any increase — they produce a flat $\pm\sqrt{0.1} = 0.316$ regardless of where we predict. The code below generates this plot:

```python
import numpy as np
import matplotlib.pyplot as plt

# Training data
x_train = np.array([-1., 0., 1.])
y_train = np.array([0.5, 1.2, 1.8])
Phi = np.column_stack([np.ones(3), x_train])

# Hyperparameters
alpha, beta = 1.0, 10.0

# Posterior
S_inv = alpha * np.eye(2) + beta * Phi.T @ Phi   # precision matrix
S_N = np.linalg.inv(S_inv)                        # posterior covariance
m_N = beta * S_N @ Phi.T @ y_train                # posterior mean (= MAP estimate)

# Predict over a grid
x_grid = np.linspace(-2, 5, 300)
Phi_grid = np.column_stack([np.ones_like(x_grid), x_grid])

pred_mean = Phi_grid @ m_N
param_var = np.array([phi @ S_N @ phi for phi in Phi_grid])
total_std  = np.sqrt(1/beta + param_var)

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x_grid, pred_mean, color='steelblue', lw=2, label='Predictive mean')
ax.fill_between(x_grid, pred_mean - total_std, pred_mean + total_std,
                alpha=0.25, color='steelblue', label='±1σ (noise + param. uncert.)')
ax.fill_between(x_grid, pred_mean - np.sqrt(1/beta), pred_mean + np.sqrt(1/beta),
                alpha=0.4, color='orange', label='±1σ (noise only, MLE/MAP)')
ax.scatter(x_train, y_train, zorder=5, color='black', s=60, label='Training data')
ax.axvspan(-1, 1, alpha=0.06, color='gray', label='Training range')
ax.set_xlabel("$x_*$"); ax.set_ylabel("$y_*$")
ax.set_title('Bayesian predictive uncertainty fans out with extrapolation')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('bayesian_uncertainty_bands.png', dpi=120, bbox_inches='tight')
```

[FIG:ORIGINAL — Bayesian linear regression predictive uncertainty: fitted line with two uncertainty bands overlaid — a narrow constant band (MLE/MAP, noise only) and a wider fan-shaped band (Bayesian, noise + parameter uncertainty) that is narrow within the training range x ∈ [-1,1] and widens substantially for extrapolation to x > 2. Training data shown as scatter points. Demonstrates the key advantage of the Bayesian predictive distribution over point-estimate approaches.]

**Why does variance grow?** The slope $w_1$ has posterior standard deviation $\sqrt{1/21} \approx 0.22$. When predicting at $x_* = 4$, a $\pm 0.22$ uncertainty in the slope translates to a $\pm 0.88$ uncertainty in the prediction — just from the slope alone. This is the extrapolation risk that linear regression practitioners intuitively fear, now made quantitative.

**Beyond the reading:** The predictive variance decomposition — $\sigma^2_{\text{pred}} = \underbrace{\beta^{-1}}_{\text{aleatoric}} + \underbrace{\boldsymbol{\phi}_*^T \mathbf{S}_N \boldsymbol{\phi}_*}_{\text{epistemic}}$ — is the closed-form, tractable instance of a much more general idea. *Aleatoric uncertainty* is irreducible noise inherent to the target itself (e.g., a user segment with genuinely variable spend, regardless of how much data you have). *Epistemic uncertainty* is reducible uncertainty from not having seen enough data to pin down the model parameters (e.g., a rarely-observed user type for which the model's weights are poorly constrained). In Bayesian linear regression, both fall out analytically. For neural networks — where the posterior over weights is intractable — recovering this same decomposition requires approximate methods: MC Dropout, Deep Ensembles, or heteroscedastic output heads for the aleatoric component. The conceptual split is identical; only the machinery changes.

**Section wrap-up.** The Bayesian posterior over **w** is a Gaussian with mean equal to the MAP estimate and covariance $\mathbf{S}_N$ that shrinks as more data arrives. MAP discards $\mathbf{S}_N$; the Bayesian approach carries it through to predictions. The payoff is a predictive variance that honestly reflects both irreducible noise and parameter uncertainty — growing where the model genuinely doesn't know, and tight where data is dense. This is the core advantage of the Bayesian approach over point estimates.

---

## 6. Advantages, Limitations, and When to Use Each Approach

**Roadmap.** With all three approaches now derived and compared numerically, this section steps back to assess them as engineering choices. We first summarize the Bayesian approach's strengths and weaknesses, then give practical guidelines for when each approach is appropriate, and close with the key cost-benefit tradeoff.

### 6.1 Advantages of the Bayesian Approach

**Inclusion of prior knowledge.** The prior $p(\mathbf{w})$ is a first-class object. It can encode physical constraints, results from related experiments, or domain expertise that the current dataset alone cannot provide.

**Calibrated uncertainty.** Predictive uncertainty quantifies both observation noise and parameter uncertainty. Predictions come with honest error bars that grow where the model genuinely doesn't know — in extrapolation regions or feature combinations underrepresented in training data. As shown in Section 5.5, this is not a minor correction: at $x_* = 4$, parameter uncertainty exceeds observation noise by nearly $8\times$.

### 6.2 Limitations of the Bayesian Approach

**The posterior is usually intractable.** The marginal likelihood $p(\mathcal{D}) = \int p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})\, d\mathbf{w}$ requires integrating over all of parameter space. For nonlinear models, this has no closed form. The main approximation strategies are: the **Laplace approximation** (fit a Gaussian to the posterior mode — cheap but can miss multimodality), **variational inference** (optimize a tractable approximate posterior — scalable, used in VAEs), and **MCMC** (sample from the posterior — asymptotically exact but expensive). Each trades computational cost against approximation fidelity.

**Priors are often chosen for mathematical convenience, not genuine belief.** The Gaussian prior in Bayesian linear regression is used largely because it produces a tractable Gaussian posterior (conjugacy). A practitioner claiming a Gaussian prior because they genuinely believe weights are normally distributed is often rationalizing a computational choice. This is not necessarily wrong — but it is worth being honest about. In many settings, the prior encodes a regularization bias as much as a real epistemic belief.

### 6.3 When to Use Each Approach

**Prefer MLE when:**
- $N$ is large relative to model complexity — the likelihood overwhelms any reasonable prior, and MAP $\to$ MLE asymptotically anyway.
- You genuinely have no prior knowledge and don't want to risk encoding a wrong prior.
- You want the simplest, most interpretable pipeline with no additional hyperparameters.

**Prefer MAP when:**
- Data is limited relative to model complexity (small $N$, large parameter count).
- You have genuine prior knowledge, or principled reasons to prefer regularization (e.g., you expect sparse weights → Laplace prior / LASSO).
- You are seeing overfitting symptoms and want regularization with a probabilistic interpretation for the choice of $\lambda$.

**Prefer the full Bayesian approach when:**
- You need calibrated uncertainty estimates — not just point predictions but honest error bars, especially in extrapolation regions.
- The model will be used for decision-making under uncertainty, where overconfidence has real costs.
- You are doing model comparison (e.g., selecting polynomial degree) and want a principled criterion beyond cross-validation — the marginal likelihood provides this.
- The model is conjugate (e.g., Bayesian linear regression, Gaussian processes) so the posterior is tractable.

**Section wrap-up.** MLE is fast and requires no prior design; MAP adds one hyperparameter ($\lambda$, or equivalently $\tau^2$) but still runs as fast; full Bayesian inference may require expensive approximations and requires careful prior design. The gains — calibrated uncertainty and honest extrapolation behavior — are most pronounced in small-$N$, high-complexity settings. The three approaches form a spectrum from simplicity to expressiveness: MLE ignores parameter uncertainty entirely, MAP uses it to regularize but then discards it, and the Bayesian approach carries it through to every prediction.

---

## 7. The Three Approaches: A Final Comparison

At prediction time, the three approaches produce:

$$\underbrace{p(y_* \mid \mathbf{x}_*, \hat{\mathbf{w}}_{\text{MLE}})}_{\text{plug in OLS estimate}} \qquad \underbrace{p(y_* \mid \mathbf{x}_*, \hat{\mathbf{w}}_{\text{MAP}})}_{\text{plug in ridge estimate}} \qquad \underbrace{\int p(y_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w}}_{\text{average over posterior}}$$

For the running example at $x_* = 2$ (mild extrapolation):

| Approach | Predicted mean | Predicted std. dev. |
|---|---|---|
| MLE | $2.467$ | $\sqrt{0.1} = 0.316$ (noise only) |
| MAP | $2.367$ | $\sqrt{0.1} = 0.316$ (noise only) |
| Bayesian | $2.367$ | $\sqrt{0.323} = 0.568$ (noise + param. uncert.) |

MLE and MAP give identical, constant uncertainty regardless of where we predict. Bayesian inference produces larger, more honest uncertainty at this extrapolation point — and smaller uncertainty near the training center. As $N \to \infty$, the posterior concentrates, $\boldsymbol{\phi}_*^T \mathbf{S}_N \boldsymbol{\phi}_* \to 0$, and all three approaches converge. The Bayesian approach earns its cost in the small-$N$, high-complexity regime where uncertainty about **w** is the dominant source of prediction error.

**Document wrap-up.** The three approaches — MLE, MAP, and full Bayesian inference — are not competing philosophies but points on a single continuum. MLE maximizes the likelihood alone; MAP adds a prior term to that objective; the Bayesian approach keeps the entire posterior that MAP collapses to a point. The algebraic thread connecting them is clean: MLE = OLS, MAP = ridge (or LASSO), and the Bayesian posterior mean equals the MAP estimate with the posterior covariance as a bonus. The practical thread is equally clear: when data is plentiful, all three converge and MLE's simplicity wins; when data is scarce, the prior matters and honest uncertainty quantification — the Bayesian approach's defining advantage — becomes the difference between a prediction you can trust and one you cannot.

---

## Sources and Further Reading

- Lecture slides: "Machine Learning 1 — Bayesian Approach" (slides 2–6). Source of the core MLE/MAP/Bayesian framing, posterior and predictive distribution formulas, and curve fitting setup.
- Bishop, C. M. *Pattern Recognition and Machine Learning* (2006). Chapter 1.2 (Bayesian inference); Chapter 2.3.3 (linear Gaussian marginalisation identity); Chapter 3 (Bayesian linear regression — the posterior derivation follows Bishop §3.3, the predictive variance decomposition is equation 3.59).
- Murphy, K. P. *Machine Learning: A Probabilistic Perspective* (2012). Chapter 3 (Bayesian estimation), Chapter 7 (linear regression). Especially clear on the prior-as-regularizer connection.
- Tibshirani, R. (1996). "Regression shrinkage and selection via the lasso." *Journal of the Royal Statistical Society, Series B*, 58(1):267–288. Original LASSO paper; the MAP/Laplace prior connection is expanded in subsequent literature.
