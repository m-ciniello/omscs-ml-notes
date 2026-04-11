# The Bias-Variance Tradeoff

---

## 1. The Core Question: What Makes a Predictor Good — and What Can Go Wrong?

Suppose we have a regression model $y(\mathbf{x})$ that predicts a target $t$ given an input $\mathbf{x}$. We measure prediction quality by the squared loss $L(t, y(\mathbf{x})) = (t - y(\mathbf{x}))^2$. But a single prediction being off by some amount tells us nothing general — we care about performance *on average*, over all inputs and all possible target values drawn from the true data-generating process $p(\mathbf{x}, t)$.

That average performance is the **expected loss**:

$$\mathbb{E}[L] = \iint (t - y(\mathbf{x}))^2\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt$$

This is the fundamental quantity. But it conceals two very different sources of error — one we can eliminate by choosing a better model, and one we cannot eliminate at all. Teasing these apart is the first step. Once we do, a second subtlety emerges: in practice, we never have access to the true distribution $p(\mathbf{x}, t)$; we only have a finite dataset $\mathcal{D}$. Training on different finite datasets produces different predictors, and those predictors scatter around some average behaviour. This scattering introduces yet another source of error — one governed by the complexity of the model and the size of the dataset.

The **bias-variance decomposition** makes all of this precise. It breaks expected loss into three terms — bias, variance, and noise — each with a clear meaning and distinct implications for model design. The decomposition reveals a fundamental tension: reducing one source of error typically increases another, and the best achievable performance requires balancing them.

---

## 2. Expected Loss and the Optimal Predictor

Before worrying about what can go wrong with a trained model, we need to understand what the *best possible* predictor looks like under squared loss. This optimal predictor sets the theoretical ceiling — everything else is measured against it.

### 2.1 Setup: Regression Under Squared Loss

We assume data is drawn from some joint distribution $p(\mathbf{x}, t)$. For a given input $\mathbf{x}$, the target $t$ is not deterministic: it is drawn from the conditional $p(t \mid \mathbf{x})$. In the classic example, the true relationship might be $t = \sin(2\pi x) + \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, \beta^{-1})$ — the target is the underlying signal plus irreducible noise.

[FIG:ORIGINAL — Joint distribution p(x,t) for the sinusoidal regression example, showing the conditional p(t|x) as a Gaussian slice at a fixed x, the underlying true function sin(2πx), the model prediction y(x), and the residual between y(x) and the conditional mean E[t|x]. Similar to Bishop Figure 1.28.]

The **regression loss function** for a prediction $y(\mathbf{x})$ at a given point $(\mathbf{x}, t)$ is:

$$L(t, y(\mathbf{x})) = (t - y(\mathbf{x}))^2$$

The **expected loss** averages this over the entire data-generating distribution:

$$\mathbb{E}[L] = \iint (t - y(\mathbf{x}))^2\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt$$

### 2.2 Deriving the Optimal Predictor

What function $y(\mathbf{x})$ minimizes this expected loss? Since $y(\mathbf{x})$ is a function of $\mathbf{x}$ alone (we choose our prediction before seeing $t$), we can minimise the expected loss *pointwise* — for each $\mathbf{x}$ separately. Using $p(\mathbf{x}, t) = p(t \mid \mathbf{x})\, p(\mathbf{x})$ and noting that $p(\mathbf{x}) > 0$ everywhere in the support, minimizing the outer integral over $\mathbf{x}$ reduces to minimizing the inner conditional expectation at each $\mathbf{x}$:

$$\min_{y(\mathbf{x})} \mathbb{E}[L] \iff \min_{y(\mathbf{x})} \int (t - y(\mathbf{x}))^2\, p(t \mid \mathbf{x})\, dt \quad \text{for each } \mathbf{x}$$

This inner quantity, $\mathbb{E}_t[L(t, y(\mathbf{x})) \mid \mathbf{x}] = \int (t - y(\mathbf{x}))^2\, p(t \mid \mathbf{x})\, dt$, is the expected loss *at a fixed input* $\mathbf{x}$. To find the minimizer, differentiate with respect to $y(\mathbf{x})$ and set the result to zero. Since $y(\mathbf{x})$ is a constant with respect to the integration over $t$, we can differentiate under the integral:

$$\frac{\partial}{\partial y(\mathbf{x})} \int (t - y(\mathbf{x}))^2\, p(t \mid \mathbf{x})\, dt = \int \frac{\partial}{\partial y(\mathbf{x})} (t - y(\mathbf{x}))^2\, p(t \mid \mathbf{x})\, dt$$

Applying the chain rule: $\frac{\partial}{\partial y}(t - y)^2 = -2(t - y)$. So:

$$= -2 \int (t - y(\mathbf{x}))\, p(t \mid \mathbf{x})\, dt = 0$$

Expanding and solving:

$$\int t\, p(t \mid \mathbf{x})\, dt = y(\mathbf{x}) \int p(t \mid \mathbf{x})\, dt$$

The left side is $\mathbb{E}[t \mid \mathbf{x}]$ by definition. The right side is $y(\mathbf{x}) \cdot 1$ since the conditional density integrates to one. Therefore:

$$\boxed{y^*(\mathbf{x}) = \mathbb{E}[t \mid \mathbf{x}]}$$

**Result: the optimal predictor under squared loss is the conditional mean.** This function $\mathbb{E}[t \mid \mathbf{x}]$ is called the **regression function**. It is the best any model could possibly do — it minimizes expected loss over all possible functions $y(\mathbf{x})$, with no restrictions on complexity. In our sinusoidal example, $\mathbb{E}[t \mid x] = \sin(2\pi x)$, since the noise $\varepsilon$ is zero-mean. Any deviation of our predictor from the conditional mean is error we could, in principle, avoid.

The second-order condition confirms this is a minimum: $\frac{\partial^2}{\partial y^2}\int (t - y)^2 p(t \mid \mathbf{x})\,dt = 2\int p(t \mid \mathbf{x})\,dt = 2 > 0$.

### 2.3 Why This Matters

This result is the anchor for everything that follows. It tells us that the *entire job* of a regression model, under squared loss, is to estimate the conditional mean $\mathbb{E}[t \mid \mathbf{x}]$. Any error in our predictor can be measured as a deviation from this function. But in practice, we face two problems: (1) $\mathbb{E}[t \mid \mathbf{x}]$ is unknown — we never observe the true conditional distribution, only samples from it, and (2) even if we could recover $\mathbb{E}[t \mid \mathbf{x}]$ exactly, there would still be error from the inherent noise in the data. The next section formalizes exactly how these two sources contribute to the expected loss.

---

## 3. Decomposing Expected Loss: What We Can Control and What We Cannot

With the optimal predictor $y^*(\mathbf{x}) = \mathbb{E}[t \mid \mathbf{x}]$ in hand, we can now decompose the expected loss of *any* predictor $y(\mathbf{x})$ into a part that measures how far $y$ is from the optimum, and a part that no predictor can eliminate.

### 3.1 The Add-and-Subtract Trick

The key move is to insert and subtract the optimal predictor inside the squared loss. Starting from the expected loss:

$$\mathbb{E}[L] = \iint (y(\mathbf{x}) - t)^2\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt$$

We add and subtract $\mathbb{E}[t \mid \mathbf{x}]$ inside the square:

$$= \iint \big(y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}] + \mathbb{E}[t \mid \mathbf{x}] - t\big)^2\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt$$

Let $A = y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]$ and $B = \mathbb{E}[t \mid \mathbf{x}] - t$. Expanding $(A + B)^2 = A^2 + 2AB + B^2$:

$$= \iint A^2\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt + 2\iint AB\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt + \iint B^2\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt$$

### 3.2 The Cross-Term Vanishes

The cross-term is $2\iint (y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}])(\mathbb{E}[t \mid \mathbf{x}] - t)\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt$. Consider the inner integral over $t$ at a fixed $\mathbf{x}$. The factor $(y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}])$ does not depend on $t$ (both $y(\mathbf{x})$ and $\mathbb{E}[t \mid \mathbf{x}]$ are functions of $\mathbf{x}$ alone), so it pulls out:

$$(y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]) \int (\mathbb{E}[t \mid \mathbf{x}] - t)\, p(t \mid \mathbf{x})\, dt$$

The remaining integral is:

$$\int \mathbb{E}[t \mid \mathbf{x}]\, p(t \mid \mathbf{x})\, dt - \int t\, p(t \mid \mathbf{x})\, dt = \mathbb{E}[t \mid \mathbf{x}] \cdot 1 - \mathbb{E}[t \mid \mathbf{x}] = 0$$

The cross-term vanishes because $\mathbb{E}[t \mid \mathbf{x}]$ is, by definition, the mean of $t$ under $p(t \mid \mathbf{x})$ — so $t - \mathbb{E}[t \mid \mathbf{x}]$ has zero expected value. This is a Pythagorean-theorem-style orthogonality: the deviation of our predictor from the optimal one is "perpendicular" (in the $L^2$ sense) to the noise.

### 3.3 The Two-Term Decomposition

With the cross-term gone, we are left with the $A^2$ and $B^2$ integrals. Each simplifies by integrating out $t$.

**The $A^2$ term.** Since $A = y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]$ does not depend on $t$:

$$\iint A^2\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt = \int \big(y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]\big)^2 \underbrace{\left[\int p(t \mid \mathbf{x})\, dt\right]}_{= 1}\, p(\mathbf{x})\, d\mathbf{x} = \int \big(y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]\big)^2\, p(\mathbf{x})\, d\mathbf{x}$$

We factored $p(\mathbf{x}, t) = p(t \mid \mathbf{x})\, p(\mathbf{x})$, then used the fact that $A^2$ is constant in $t$ to collapse the inner integral to 1.

**The $B^2$ term.** Here $B = \mathbb{E}[t \mid \mathbf{x}] - t$ *does* depend on $t$, so the inner integral does not collapse trivially:

$$\iint B^2\, p(\mathbf{x}, t)\, d\mathbf{x}\, dt = \int \left[\int \big(\mathbb{E}[t \mid \mathbf{x}] - t\big)^2\, p(t \mid \mathbf{x})\, dt\right] p(\mathbf{x})\, d\mathbf{x}$$

The bracketed inner integral is exactly the definition of the conditional variance $\text{var}[t \mid \mathbf{x}] = \mathbb{E}_t\big[(t - \mathbb{E}[t \mid \mathbf{x}])^2 \mid \mathbf{x}\big]$. So the $B^2$ term becomes $\int \text{var}[t \mid \mathbf{x}]\, p(\mathbf{x})\, d\mathbf{x}$.

Combining:

$$\boxed{\mathbb{E}[L] = \int \big(y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]\big)^2 p(\mathbf{x})\, d\mathbf{x} \;+\; \int \text{var}[t \mid \mathbf{x}]\, p(\mathbf{x})\, d\mathbf{x}}$$

Each term deserves careful interpretation:

| Term | Expression | Meaning |
|------|-----------|---------|
| **Model error** | $\int (y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}])^2\, p(\mathbf{x})\, d\mathbf{x}$ | How far our predictor deviates from the best possible predictor, averaged over inputs |
| **Intrinsic noise** | $\int \text{var}[t \mid \mathbf{x}]\, p(\mathbf{x})\, d\mathbf{x}$ | Irreducible randomness in the target — the variance of $t$ around its conditional mean |

The first term is zero if and only if $y(\mathbf{x}) = \mathbb{E}[t \mid \mathbf{x}]$ everywhere. In principle, with infinite data and a sufficiently flexible model, we could drive it to zero. The second term is a property of the data-generating process itself — it cannot be reduced by any choice of $y$. In our sinusoidal example with $\varepsilon \sim \mathcal{N}(0, \beta^{-1})$, the intrinsic noise is $\beta^{-1}$ at every $x$, giving a total noise contribution of $\beta^{-1}$.

This decomposition tells us exactly what a model can and cannot achieve: even the perfect predictor has expected loss equal to the intrinsic noise. All remaining error comes from the model failing to match the regression function. But this raises the next question: in practice, we don't know $\mathbb{E}[t \mid \mathbf{x}]$. We estimate a predictor from a finite dataset. What does the model error term look like when we account for the randomness of the training data?

---

## 4. The Frequentist Problem: Finite Data and Dataset-Dependent Predictions

The decomposition in Section 3 treats $y(\mathbf{x})$ as a fixed function. In reality, $y(\mathbf{x})$ is *learned* from data — and different training datasets produce different predictors. This section sets up the frequentist framework for reasoning about this randomness, which leads directly to the bias-variance decomposition.

### 4.1 From One Predictor to an Ensemble of Predictors

Recall that the optimal predictor is $y^*(\mathbf{x}) = \mathbb{E}[t \mid \mathbf{x}]$, which is unknown. In the frequentist framework, we observe a single finite dataset $\mathcal{D} = \{(\mathbf{x}_1, t_1), \ldots, (\mathbf{x}_N, t_N)\}$ drawn i.i.d. from $p(\mathbf{x}, t)$. Using a learning algorithm (say, regularized least squares with basis functions), we fit a predictor $y_{\mathcal{D}}(\mathbf{x})$ — a function that depends on the particular dataset we happened to see.

The notation $y_{\mathcal{D}}(\mathbf{x})$ is crucial: the subscript $\mathcal{D}$ emphasizes that the learned function is a random variable (over the randomness in the training set). A different draw of $N$ points from the same distribution would produce a different $y_{\mathcal{D}}(\mathbf{x})$.

### 4.2 Averaging Over Datasets

To evaluate the learning algorithm (as opposed to any single model it produces), we consider how it performs *on average over all possible training sets* of size $N$. This means taking an expectation over $\mathcal{D}$:

$$\mathbb{E}_{\mathcal{D}}\big[(y_{\mathcal{D}}(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}])^2\big]$$

This is the model error term from Section 3, but now with $y_{\mathcal{D}}(\mathbf{x})$ replacing the fixed $y(\mathbf{x})$, and averaged over the randomness of the training data.

**The thought experiment.** Imagine generating $L$ independent training datasets, each of size $N$, from the same distribution. Each one produces a different predictor $y^{(1)}(\mathbf{x}), y^{(2)}(\mathbf{x}), \ldots, y^{(L)}(\mathbf{x})$. We can compute the average predictor:

$$\bar{y}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}[y_{\mathcal{D}}(\mathbf{x})] = \frac{1}{L}\sum_{\ell=1}^{L} y^{(\ell)}(\mathbf{x}) \quad \text{(in the limit } L \to \infty\text{)}$$

This average predictor $\bar{y}(\mathbf{x})$ is the "expected model" — what our learning algorithm produces on average. Two questions now arise naturally:

1. **Is $\bar{y}(\mathbf{x})$ close to $\mathbb{E}[t \mid \mathbf{x}]$?** If not, the algorithm has a systematic tendency to miss the true regression function, regardless of which dataset it sees. This is **bias**.
2. **How much do the individual $y^{(\ell)}(\mathbf{x})$ scatter around $\bar{y}(\mathbf{x})$?** If they scatter a lot, the algorithm is overly sensitive to the particular training set. This is **variance**.

The next section makes this precise.

---

## 5. The Bias-Variance Decomposition

This is the central result. We decompose the expected model error — the gap between our learned predictor and the optimal predictor, averaged over training sets — into two terms with competing behaviours. The derivation follows the same add-and-subtract strategy used in Section 3, now applied over the distribution of datasets.

### 5.1 Derivation

Start from the expected squared deviation of the learned predictor from the regression function, integrated over input space (this is the model error term from Section 3, now with the dataset expectation applied):

$$\mathbb{E}_{\mathcal{D}}\left[\int \big(y_{\mathcal{D}}(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]\big)^2\, p(\mathbf{x})\, d\mathbf{x}\right]$$

Exchanging the order of integration and expectation (by Fubini's theorem, since all terms are non-negative), we can work pointwise at each $\mathbf{x}$ and focus on the inner quantity:

$$\mathbb{E}_{\mathcal{D}}\big[(y_{\mathcal{D}}(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}])^2\big]$$

Now add and subtract the expected predictor $\mathbb{E}_{\mathcal{D}}[y_{\mathcal{D}}(\mathbf{x})] = \bar{y}(\mathbf{x})$:

$$= \mathbb{E}_{\mathcal{D}}\big[\big(y_{\mathcal{D}}(\mathbf{x}) - \bar{y}(\mathbf{x}) + \bar{y}(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]\big)^2\big]$$

Expanding the square — with $A = y_{\mathcal{D}}(\mathbf{x}) - \bar{y}(\mathbf{x})$ and $B = \bar{y}(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]$:

$$= \mathbb{E}_{\mathcal{D}}[A^2] + 2\,\mathbb{E}_{\mathcal{D}}[AB] + B^2$$

Note that $B$ does not depend on $\mathcal{D}$ (it involves only the fixed quantities $\bar{y}(\mathbf{x})$ and $\mathbb{E}[t \mid \mathbf{x}]$), so $\mathbb{E}_{\mathcal{D}}[B^2] = B^2$.

**The cross-term vanishes:** $\mathbb{E}_{\mathcal{D}}[AB] = B \cdot \mathbb{E}_{\mathcal{D}}[A] = B \cdot \mathbb{E}_{\mathcal{D}}[y_{\mathcal{D}}(\mathbf{x}) - \bar{y}(\mathbf{x})] = B \cdot (\bar{y}(\mathbf{x}) - \bar{y}(\mathbf{x})) = 0$. The cross-term vanishes because $\bar{y}(\mathbf{x})$ is, by definition, the mean of $y_{\mathcal{D}}(\mathbf{x})$ over datasets — so the deviation from that mean has zero expectation.

We are left with:

$$\mathbb{E}_{\mathcal{D}}\big[(y_{\mathcal{D}}(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}])^2\big] = \underbrace{\mathbb{E}_{\mathcal{D}}\big[(y_{\mathcal{D}}(\mathbf{x}) - \bar{y}(\mathbf{x}))^2\big]}_{\text{variance}} + \underbrace{\big(\bar{y}(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]\big)^2}_{\text{(bias)}^2}$$

### 5.2 The Full Three-Term Decomposition

We can now assemble the full result. In Section 3.3, we showed that the expected loss of any predictor $y(\mathbf{x})$ decomposes as:

$$\mathbb{E}[L] = \int \big(y(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]\big)^2\, p(\mathbf{x})\, d\mathbf{x} + \int \text{var}[t \mid \mathbf{x}]\, p(\mathbf{x})\, d\mathbf{x}$$

The second term (noise) does not depend on the predictor or the training data — it is a constant. The first term (model error) is what we just decomposed in Section 5.1. Substituting $y(\mathbf{x}) = y_{\mathcal{D}}(\mathbf{x})$ and taking $\mathbb{E}_{\mathcal{D}}$ of both sides (only the model error term depends on $\mathcal{D}$):

$$\mathbb{E}_{\mathcal{D}}[\mathbb{E}[L]] = \int \underbrace{\mathbb{E}_{\mathcal{D}}\big[(y_{\mathcal{D}}(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}])^2\big]}_{= \text{variance}(\mathbf{x}) + (\text{bias}(\mathbf{x}))^2}\, p(\mathbf{x})\, d\mathbf{x} + \int \text{var}[t \mid \mathbf{x}]\, p(\mathbf{x})\, d\mathbf{x}$$

The model error at each $\mathbf{x}$ splits into bias$^2$ + variance (Section 5.1). Distributing the integral gives the expected loss of a learning algorithm, averaged over both test data and training data:

$$\boxed{\mathbb{E}\big[\mathbb{E}_{\mathcal{D}}[L]\big] = \underbrace{\int \big(\bar{y}(\mathbf{x}) - \mathbb{E}[t \mid \mathbf{x}]\big)^2 p(\mathbf{x})\, d\mathbf{x}}_{\text{(bias)}^2} + \underbrace{\int \mathbb{E}_{\mathcal{D}}\big[(y_{\mathcal{D}}(\mathbf{x}) - \bar{y}(\mathbf{x}))^2\big]\, p(\mathbf{x})\, d\mathbf{x}}_{\text{variance}} + \underbrace{\int \text{var}[t \mid \mathbf{x}]\, p(\mathbf{x})\, d\mathbf{x}}_{\text{noise}}}$$

### 5.3 What Each Term Means

**Bias squared** measures the systematic error of the learning algorithm — how far the *average* predictor (averaged over all possible training sets) deviates from the true regression function. High bias means the model class is too restrictive to capture the true relationship, regardless of how much data it sees. A constant predictor has high bias for any non-trivial target function; a very flexible model class (e.g., high-degree polynomial) has low bias because, on average over many datasets, its predictions can track the true function.

**Variance** measures the sensitivity of the learned predictor to the particular training set. High variance means that small changes in the training data cause large changes in the learned function — the model is fitting noise rather than signal. Flexible models (many parameters relative to $N$) have high variance because they have enough capacity to memorize the idiosyncrasies of each particular dataset. Rigid models have low variance because their predictions change little regardless of which dataset they happen to see.

**Noise** is the irreducible error — the inherent stochasticity of the target variable. No model, no matter how complex or how much data it has, can reduce this term. It sets an absolute lower bound on the expected loss.

The intuition is best summarized as a table of the two extremes:

| Regime | Bias | Variance | Character |
|--------|------|----------|-----------|
| **Underfitting** (model too simple) | High | Low | The model cannot capture the true pattern — it makes the same systematic error on every training set |
| **Overfitting** (model too complex) | Low | High | The model captures the true pattern *plus* the noise — each training set produces a wildly different predictor |

The optimal model complexity sits between these extremes, at the point where the sum bias$^2$ + variance is minimized.

### 5.4 Why Bias and Variance Trade Off

The tradeoff is not a mathematical inevitability — it is an empirical regularity with a clear mechanistic explanation. When you increase model complexity (e.g., use more basis functions, reduce regularization):

- **Bias decreases** because a richer function class can represent the true regression function more closely, so the average predictor over datasets gets closer to $\mathbb{E}[t \mid \mathbf{x}]$.
- **Variance increases** because the richer function class has more degrees of freedom that can be driven by noise in any particular dataset. Each individual predictor $y_{\mathcal{D}}(\mathbf{x})$ is more volatile.

Conversely, reducing complexity (e.g., using fewer basis functions, increasing regularization) increases bias and decreases variance. The expected loss — their sum plus noise — typically has a U-shape as a function of complexity, with a minimum at some intermediate value.

There are rare cases where both bias and variance can be reduced simultaneously (e.g., ensemble methods like bagging reduce variance without increasing bias, and boosting reduces bias without increasing variance much). But within a single model class parameterized by a complexity parameter, the tradeoff holds.

### 5.5 A Tiny Numerical Example

To make the decomposition tangible, consider the simplest possible case. We have a true regression function $h(x) = 2x$ with noise variance $\sigma^2 = 1$. We train a constant model $y_{\mathcal{D}} = c$ (the sample mean of the targets) on datasets of size $N = 1$, and evaluate at a single point $x = 3$.

With $N = 1$, each "dataset" is a single observation $t = 2x_{\text{train}} + \varepsilon$, and our constant model just outputs $y_{\mathcal{D}} = t$. To keep things concrete, suppose the training input is always $x_{\text{train}} = 1$ (so we are studying the algorithm's behaviour at a fixed training location). Then $y_{\mathcal{D}} = 2 + \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, 1)$.

At the test point $x = 3$, the true regression value is $h(3) = 6$.

**Average predictor:** $\bar{y} = \mathbb{E}_{\mathcal{D}}[y_{\mathcal{D}}] = \mathbb{E}[2 + \varepsilon] = 2$.

**Bias:** $\bar{y} - h(3) = 2 - 6 = -4$, so $(\text{bias})^2 = 16$. The constant model trained at $x=1$ is systematically wrong at $x=3$.

**Variance:** $\mathbb{E}_{\mathcal{D}}[(y_{\mathcal{D}} - \bar{y})^2] = \mathbb{E}[\varepsilon^2] = \sigma^2 = 1$. Different training datasets cause the prediction to scatter by $\pm 1$ around the mean.

**Noise:** $\text{var}[t \mid x=3] = \sigma^2 = 1$.

**Total expected loss:** $16 + 1 + 1 = 18$. We can verify directly: $\mathbb{E}[(y_{\mathcal{D}} - t_*)^2] = \mathbb{E}[(2 + \varepsilon_{\text{train}} - 6 - \varepsilon_{\text{test}})^2] = (-4)^2 + \text{Var}(\varepsilon_{\text{train}}) + \text{Var}(\varepsilon_{\text{test}}) = 16 + 1 + 1 = 18$. The decomposition checks out.

The example makes the roles vivid: the dominant error is bias (a constant model cannot represent a linear function evaluated far from its training location). Variance is small (one degree of freedom). A richer model (e.g., a line) would eliminate the bias at the cost of higher variance — but here the tradeoff clearly favours more complexity.

---

## 6. Worked Example: Regularized Basis Function Regression

The bias-variance tradeoff is abstract without a concrete demonstration. This section walks through the example from Bishop §3.5–3.6: fitting a sinusoidal truth with regularized Gaussian basis functions and varying the regularization strength $\lambda$.

### 6.1 The Setup

**True data-generating process:**

$$t = \sin(2\pi x) + \varepsilon, \qquad x \sim U(0, 1), \quad \varepsilon \sim \mathcal{N}(0, \alpha^{-1})$$

The regression function is $\mathbb{E}[t \mid x] = \sin(2\pi x)$.

**The experiment:**

1. Generate $L$ independent training datasets, each containing $N = 25$ points.
2. For each dataset $\mathcal{D}^{(\ell)}$, fit a linear model using $M = 24$ Gaussian basis functions with a regularized least-squares objective:

$$E_{\mathcal{D}} = \frac{1}{2}\sum_{n=1}^{N}\big(t_n - \mathbf{w}^T\boldsymbol{\phi}(x_n)\big)^2 + \frac{\lambda}{2}\mathbf{w}^T\mathbf{w}$$

Each dataset produces a different weight vector $\mathbf{w}^{(\ell)}$ and therefore a different predictor $y^{(\ell)}(x) = (\mathbf{w}^{(\ell)})^T\boldsymbol{\phi}(x)$.

3. Compute the empirical average predictor $\bar{y}(x) = \frac{1}{L}\sum_\ell y^{(\ell)}(x)$.

The regularization parameter $\lambda$ controls model complexity. Since we use $M = 24$ basis functions for $N = 25$ data points, the model is nearly interpolating when $\lambda$ is small. The regularizer $\frac{\lambda}{2}\mathbf{w}^T\mathbf{w}$ is Ridge regression — equivalently, MAP estimation under a Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \lambda^{-1}I)$, as derived in the MLE/MAP/Bayesian notes (Section 3.2).

### 6.2 Three Regularization Regimes

The effect of $\lambda$ is dramatically visible in the individual predictors and their average:

**Large $\lambda$ (e.g. $\ln\lambda = 2.6$) — heavy regularization, underfitting:**

The strong penalty constrains the weights to be small, forcing the predictor to be nearly flat. Every one of the $L$ individual fits $y^{(\ell)}(x)$ looks similar — they all barely deviate from zero. The average $\bar{y}(x)$ is smooth but systematically fails to capture the sine wave. **Variance is low** (the fits cluster tightly), but **bias is high** (the average is far from $\sin(2\pi x)$).

**Small $\lambda$ (e.g. $\ln\lambda = -2.4$) — light regularization, overfitting:**

With little penalty, the model is nearly free to use all 24 basis functions to chase the noise in each dataset. The individual fits $y^{(\ell)}(x)$ are wildly different from one another — some oscillate dramatically, some track the data closely in one region but diverge elsewhere. But their average $\bar{y}(x)$ is remarkably close to $\sin(2\pi x)$, because the noise-driven deviations average out. **Bias is low** (the average predictor is accurate), but **variance is high** (each individual predictor is unreliable).

**Intermediate $\lambda$ (e.g. $\ln\lambda = -0.31$) — the sweet spot:**

The regularizer is strong enough to suppress noise-driven oscillations but weak enough to let the model capture the sine wave. The individual fits $y^{(\ell)}(x)$ show moderate spread around an average that tracks the truth well. **Both bias and variance are moderate**, and their sum is minimized.

[FIG:READING — Figure 3.5 from Bishop: grid of plots showing L individual predictors (red curves) on the left and the average predictor $\bar{y}(x)$ (green) vs. truth (dark) on the right, for three values of $\ln\lambda$. Top row: $\ln\lambda = 2.6$ (underfitting); middle: $\ln\lambda = -0.31$ (balanced); bottom: $\ln\lambda = -2.4$ (overfitting).]

### 6.3 Estimating Bias and Variance Empirically

Given $L$ datasets, $N$ evaluation points $\{x_1, \ldots, x_N\}$ (drawn from $p(x)$), and $L$ fitted predictors $y^{(1)}, \ldots, y^{(L)}$, we can estimate the bias and variance terms numerically.

**Average predictor at each evaluation point:**

$$\bar{y}(x_n) = \frac{1}{L}\sum_{\ell=1}^{L} y^{(\ell)}(x_n)$$

**Bias squared** (approximating the integral $\int (\bar{y}(x) - \mathbb{E}[t \mid x])^2\, p(x)\, dx$ by a sample average):

$$(\text{bias})^2 \approx \frac{1}{N}\sum_{n=1}^{N}\big(\bar{y}(x_n) - \mathbb{E}[t \mid x_n]\big)^2$$

In our example, $\mathbb{E}[t \mid x_n] = \sin(2\pi x_n)$ is known, so this can be computed exactly.

**Variance** (approximating $\int \mathbb{E}_{\mathcal{D}}[(y_{\mathcal{D}}(x) - \bar{y}(x))^2]\, p(x)\, dx$):

$$\text{variance} \approx \frac{1}{N}\sum_{n=1}^{N} \frac{1}{L}\sum_{\ell=1}^{L}\big(y^{(\ell)}(x_n) - \bar{y}(x_n)\big)^2$$

The outer average is over evaluation points (approximating the integral over $p(x)$); the inner average is over datasets (approximating $\mathbb{E}_{\mathcal{D}}$).

### 6.4 The Tradeoff Curve

Plotting bias$^2$, variance, and their sum as functions of $\ln\lambda$ reveals the classic U-shaped tradeoff:

[FIG:READING — Figure 3.6 from Bishop: bias-variance decomposition plot showing (bias)² (blue), variance (red), (bias)² + variance (magenta), and test error (black) as functions of $\ln\lambda$. The test error has a U-shape with minimum at an intermediate $\lambda$.]

Reading the plot from left to right ($\ln\lambda$ increasing, i.e., regularization getting stronger):

- **Left side** ($\ln\lambda \ll 0$, weak regularization): bias is low but variance is high. The model is flexible enough to track the truth on average, but each individual fit is noisy. This is the **overfitting regime**.
- **Right side** ($\ln\lambda \gg 0$, strong regularization): variance is low but bias is high. The model is too rigid to follow the truth, though it is consistent across datasets. This is the **underfitting regime**.
- **Minimum of the sum**: the optimal $\lambda$ balances bias and variance. The test error (black curve) closely tracks (bias)$^2$ + variance, offset by the constant noise floor.

The gap between the test error curve and the (bias)$^2$ + variance curve is exactly the noise term — a constant that does not depend on $\lambda$. This confirms the decomposition empirically.

### 6.5 A Numerical Demonstration

The following code runs the full experiment: generates $L$ datasets from the sinusoidal truth, fits regularized basis function models for a range of $\lambda$ values, and plots the bias-variance decomposition.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
L, N, M = 200, 25, 24
noise_std = 0.3
x_eval = np.linspace(0, 1, 200)
truth = np.sin(2 * np.pi * x_eval)

# Gaussian basis functions, evenly spaced
centres = np.linspace(0, 1, M)
scale = 1.0 / M

def basis(x):
    return np.exp(-0.5 * ((x[:, None] - centres[None, :]) / scale) ** 2)

Phi_eval = basis(x_eval)
ln_lambdas = np.linspace(-3, 2, 60)

bias_sq_all, var_all, mse_all = [], [], []

for ln_lam in ln_lambdas:
    lam = np.exp(ln_lam)
    preds = np.zeros((L, len(x_eval)))

    for ell in range(L):
        x_train = rng.uniform(0, 1, N)
        t_train = np.sin(2 * np.pi * x_train) + rng.normal(0, noise_std, N)
        Phi_train = basis(x_train)
        # Ridge regression closed-form solution
        w = np.linalg.solve(Phi_train.T @ Phi_train + lam * np.eye(M),
                            Phi_train.T @ t_train)
        preds[ell] = Phi_eval @ w

    y_bar = preds.mean(axis=0)
    bias_sq = np.mean((y_bar - truth) ** 2)
    variance = np.mean(np.mean((preds - y_bar[None, :]) ** 2, axis=0))
    mse = np.mean(np.mean((preds - truth[None, :]) ** 2, axis=0))

    bias_sq_all.append(bias_sq)
    var_all.append(variance)
    mse_all.append(mse)

bias_sq_all = np.array(bias_sq_all)
var_all = np.array(var_all)
mse_all = np.array(mse_all)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ln_lambdas, bias_sq_all, color='royalblue', lw=2, label=r'$(\mathrm{bias})^2$')
ax.plot(ln_lambdas, var_all, color='crimson', lw=2, label='variance')
ax.plot(ln_lambdas, bias_sq_all + var_all, color='magenta', lw=2,
        label=r'$(\mathrm{bias})^2$ + variance')
ax.plot(ln_lambdas, mse_all, color='black', lw=2, label='test error (MSE)')
ax.axhline(noise_std**2, color='gray', ls=':', lw=1, label=f'noise floor ($\\sigma^2={noise_std**2}$)')
ax.set_xlabel(r'$\ln\lambda$')
ax.set_ylabel('Error')
ax.set_title('Bias-variance decomposition for regularized basis function regression')
ax.legend()
ax.set_ylim(bottom=0, top=0.25)
plt.tight_layout()
plt.savefig('bias_variance_tradeoff_curve.png', dpi=150, bbox_inches='tight')
```

The experiment confirms the decomposition empirically and makes the tradeoff visible: there is a single value of $\ln\lambda$ where the sum of bias$^2$ and variance is minimized, and the test error at that point sits exactly one noise floor above it. Moving $\lambda$ in either direction trades one error source for the other. The practical question — how to find this optimum without access to the true regression function — is what the next section addresses.

---

## 7. The Tradeoff in Practice: Model Complexity, Regularization, and Beyond

The bias-variance decomposition is not just a theoretical curiosity — it fundamentally shapes how we design, train, and evaluate machine learning models. This section connects the decomposition to practical modelling decisions and previews how the Bayesian framework sidesteps some of its limitations.

### 7.1 Model Complexity Is the Lever

In the worked example, the regularization parameter $\lambda$ served as a continuous dial for model complexity. But the same tradeoff appears in every modelling decision that adjusts capacity:

| Complexity lever | More complex (lower bias, higher variance) | Less complex (higher bias, lower variance) |
|---|---|---|
| Polynomial degree | Higher degree | Lower degree |
| Number of basis functions | More functions | Fewer functions |
| Regularization ($\lambda$) | Smaller $\lambda$ | Larger $\lambda$ |
| Neural network depth/width | Deeper/wider | Shallower/narrower |
| k in k-NN | Smaller $k$ (1-NN: low bias, high variance) | Larger $k$ (all-points average: high bias, low variance) |
| Tree depth (decision trees) | Deeper | Shallower |

The optimal setting always depends on $N$ (the training set size) relative to the effective model complexity. With more data, the variance term shrinks (each dataset becomes more representative of the truth), so a more complex model becomes affordable. This is why the "best" model complexity increases with dataset size — a fact that the bias-variance lens makes quantitative.

### 7.2 Why Not Just Estimate Bias and Variance Directly?

The experiment in Section 6 required generating $L$ independent datasets — a luxury we never have in practice. With a single dataset, we cannot directly estimate bias and variance separately; we can only estimate their sum (the expected test error) via cross-validation or a held-out test set.

This is a fundamental limitation of the frequentist bias-variance framework: it tells us *what* is happening (whether our error is dominated by bias or variance) only in thought experiments. In practice, we use indirect signals:

- **Training error much lower than test error** → variance-dominated (overfitting). Remedy: more data, stronger regularization, simpler model.
- **Training error and test error both high** → bias-dominated (underfitting). Remedy: more flexible model, weaker regularization, better features.
- **Training error close to test error, both moderately high** → may be near the noise floor, or both bias and variance contribute. Check by increasing model complexity — if test error doesn't improve, you are likely noise-limited.

### 7.3 Dataset Size and Optimal Complexity

An important practical consequence: the optimal model complexity is not a fixed property of the problem — it depends on $N$. With more data:

- **Variance decreases** because each training set is a better sample of the true distribution. The individual predictors $y^{(\ell)}$ cluster more tightly around $\bar{y}$.
- **Bias is unchanged** (or slightly decreases if the model class is rich enough to benefit from more data, e.g., through better nonparametric estimation).
- **The optimal complexity shifts toward more flexible models** because we can "afford" higher variance when it is inherently smaller.

This explains the empirical observation that deep neural networks — which have enormous capacity and would massively overfit on small datasets — work well when $N$ is very large. The variance is controlled not by restricting model class, but by providing enough data.

### 7.4 Ensemble Methods and the Decomposition

The bias-variance decomposition explains why ensemble methods work so well. **Bagging** (bootstrap aggregating) trains $L$ models on bootstrap resamples of the training data and averages their predictions. This is a direct, practical approximation to $\bar{y}(\mathbf{x}) = \mathbb{E}_{\mathcal{D}}[y_{\mathcal{D}}(\mathbf{x})]$ — replacing the hypothetical average over all possible datasets with an average over bootstrap datasets. Averaging reduces variance (the individual models' predictions scatter around the mean, and averaging smooths out the scatter) while leaving bias approximately unchanged (the average of biased predictors is still biased by the same amount). This is why bagging helps most with high-variance, low-bias learners like deep decision trees.

**Boosting** works the other end. At each iteration, a new weak learner is fit to the *residuals* of the current ensemble — the gap between the ensemble's predictions and the targets. Each new learner corrects the systematic errors of its predecessors, so the ensemble's average prediction $\bar{y}(\mathbf{x})$ moves closer to $\mathbb{E}[t \mid \mathbf{x}]$ with each step: this is bias reduction. Variance does increase with each added learner (more parameters that can respond to noise), but a small learning rate (shrinkage) ensures each learner contributes only a fraction of its fit, keeping the variance growth slow relative to the bias reduction. The net effect is a steady decrease in total error — until, eventually, the variance cost of additional learners exceeds the bias benefit.

### 7.5 The Bayesian Resolution

The bias-variance decomposition is a *frequentist* concept — it evaluates a learning algorithm by imagining repeated sampling of datasets. A Bayesian approach sidesteps the framework entirely by not committing to a single predictor.

Recall from the MLE/MAP/Bayesian notes (Section 4.3) that the Bayesian predictive distribution marginalizes over all possible parameter values:

$$p(t_* \mid \mathbf{x}_*, \mathcal{D}) = \int p(t_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w}$$

This is not a single predictor $y_{\mathcal{D}}(\mathbf{x})$ — it is a weighted average over all predictors consistent with the data, with weights given by the posterior. The predictive variance automatically decomposes into observation noise and parameter uncertainty, capturing what the frequentist framework calls "variance" without needing to imagine repeated datasets.

From the bias-variance perspective, the Bayesian predictive mean $\mathbb{E}[t_* \mid \mathbf{x}_*, \mathcal{D}]$ can still be analyzed for bias and variance in the frequentist sense (treating $\mathcal{D}$ as random), and it typically has good bias-variance properties: the prior acts as a regularizer (controlling variance) while the averaging over the posterior reduces the effective model complexity adaptively.

More practically, in the frequentist framework the main tool for selecting model complexity is cross-validation — splitting the data to estimate test error and choosing $\lambda$ to minimize it. The Bayesian framework offers an alternative: the **marginal likelihood** (model evidence) $p(\mathcal{D}) = \int p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})\, d\mathbf{w}$ provides a principled score for model comparison that naturally penalizes excessive complexity (Bayesian Occam's razor). This avoids the need to hold out data for validation — all data is used for inference. The tradeoff is computational: marginal likelihoods are tractable only for conjugate models like Bayesian linear regression.

The lecture slides frame Bayesian regression as the answer to the question "how do we select model complexity without splitting the data?" — and this is exactly the marginal likelihood's role. It replaces the frequentist's cross-validation loop with a single integral that balances data fit against model complexity.

---

## 8. Summary: The Three Components of Prediction Error

The bias-variance decomposition reveals that the expected loss of a learning algorithm under squared loss decomposes into three additive components:

$$\text{expected loss} = (\text{bias})^2 + \text{variance} + \text{noise}$$

| Component | Controlled by | Reducible? |
|-----------|--------------|------------|
| **Bias$^2$** — systematic deviation of the average predictor from the truth | Model flexibility (capacity, regularization) | Yes: use a more flexible model or weaker regularization |
| **Variance** — sensitivity of the predictor to the training set | Model flexibility, dataset size | Yes: use a simpler model, stronger regularization, more data, or ensembling |
| **Noise** — inherent stochasticity of the target | Nothing | No: it is a property of the data-generating process |

The fundamental tension is that bias and variance typically move in opposite directions as a function of model complexity. The art of machine learning is finding the complexity setting — or the algorithmic approach — that minimizes their sum.

---

## Sources and Further Reading

- Lecture slides: "Machine Learning 1 — Bias-Variance Decomposition" (slides 2–10). Source of the expected loss decomposition, the frequentist framework for evaluating over datasets, and the regularized basis function example.
- Bishop, C. M. *Pattern Recognition and Machine Learning* (2006). §1.5.5 (bias-variance decomposition for loss functions), §3.2 (bias-variance decomposition for regularized linear models), §3.5–3.6 (worked example with Gaussian basis functions — Figures 3.5 and 3.6). The derivation and example in this document follow Bishop's treatment closely.
- Hastie, T., Tibshirani, R., and Friedman, J. *The Elements of Statistical Learning*, 2nd ed. (2009). §2.9 (bias-variance decomposition), §7.3 (bias-variance and model selection). Particularly clear on the connection between model complexity, training set size, and optimal regularization.
- Geman, S., Bienenstock, E., and Doursat, R. (1992). "Neural networks and the bias/variance dilemma." *Neural Computation*, 4(1):1–58. The foundational paper on bias-variance in the context of flexible statistical models.
- Friedman, J. H. (1997). "On bias, variance, 0/1 loss, and the curse-of-dimensionality." *Data Mining and Knowledge Discovery*, 1(1):55–77. Extends the decomposition beyond squared loss to classification.
