# Uncertainty Quantification in Machine Learning

---

## 1. Why Predictions Need Error Bars

In the *MLE, MAP, and the Bayesian Approach* notes, we derived a clean result for Bayesian linear regression: the predictive variance decomposes as $\sigma^2_{\text{pred}}(\mathbf{x}_*) = \beta^{-1} + \boldsymbol{\phi}_*^T \mathbf{S}_N \boldsymbol{\phi}_*$ — irreducible observation noise plus a term that grows with extrapolation distance. That decomposition fell out analytically because the posterior $p(\mathbf{w} \mid \mathcal{D})$ was Gaussian in closed form.

But modern ML runs on neural networks, gradient-boosted trees, and other models where the posterior over parameters is intractable. The question this document answers is: **how do we recover honest uncertainty estimates when closed-form Bayesian inference is unavailable?**

The practical stakes are high. A classifier that outputs "80% probability of benign" is useless unless that 80% means something calibrated — if you collect all inputs that receive 80%, roughly 80% should actually be benign. A regression model that predicts "revenue = \$4.2M" without an error bar invites false confidence: is the model sure because the input resembles well-covered training data, or is it guessing because the input is far from anything it has seen?

Three failure modes recur in practice:

1. **Overconfident extrapolation.** A point-estimate model assigns the same confidence to an interpolation (input surrounded by training data) and an extrapolation (input far from any training example). The Bayesian predictive variance from the BLR notes already illustrated this: MLE/MAP produce a flat $\pm\sqrt{0.1}$ regardless of $x_*$, while the Bayesian band fans out to $\pm\sqrt{0.893}$ at $x_* = 4$.

2. **Miscalibrated probabilities.** A neural network trained with cross-entropy loss produces softmax outputs that are often interpreted as probabilities but can be systematically overconfident — modern deep networks tend to push softmax outputs toward 0 and 1 even when the true class probabilities are moderate (Guo et al. 2017).

3. **Inability to distinguish what the model doesn't know from what is inherently noisy.** A skin-lesion classifier may be uncertain because the image is ambiguous (aleatoric — even a dermatologist would be unsure) or because the lesion type is underrepresented in training data (epistemic — more data would help). These two kinds of uncertainty call for different responses: aleatoric uncertainty suggests collecting more features, while epistemic uncertainty suggests collecting more data or flagging for human review.

These notes develop the tools to address all three failure modes. We begin by formalizing the aleatoric/epistemic distinction (Section 2), then develop the three dominant approaches to epistemic uncertainty in neural networks — Bayesian neural networks as the intractable ideal (Section 3), MC Dropout (Section 4), and Deep Ensembles (Section 5). Section 6 turns to the other half: modeling input-dependent aleatoric uncertainty with heteroscedastic output heads. Section 7 addresses calibration — the post-hoc correction that ensures predicted probabilities match empirical frequencies. Section 8 introduces proper scoring rules for evaluating probabilistic predictions. Finally, Section 9 distills the preceding material into a recommended end-to-end pipeline and practical decision checklist.

---

## 2. A Taxonomy of Uncertainty: Aleatoric vs. Epistemic

Every predictive model contends with two fundamentally different kinds of uncertainty, and distinguishing them matters for both model development and deployment. This section formalizes the aleatoric/epistemic split — a generalization of the predictive variance decomposition derived for Bayesian linear regression — and shows how the two types manifest differently in practice.

### 2.1 Definitions and Intuition

**Aleatoric uncertainty** (from Latin *alea*, "dice") is irreducible randomness inherent to the data-generating process. It cannot be reduced by collecting more training data because it reflects genuine variability in the target given the inputs.

**Example:** Predicting the exact closing price of a stock from yesterday's features. Even with perfect knowledge of the relationship between features and price, market microstructure noise, trader behaviour, and unobserved information shocks introduce irreducible variance. No amount of historical data will eliminate it.

**Epistemic uncertainty** (from Greek *epistēmē*, "knowledge") is uncertainty due to limited knowledge — specifically, limited training data. It can, in principle, be reduced to zero by observing enough data to pin down the model's parameters.

**Example:** A medical imaging model trained on 50 chest X-rays. The model's predictions are uncertain partly because 50 images are insufficient to determine the millions of network weights. Train on 50,000 images and this component shrinks dramatically, even though the aleatoric noise (imaging artifacts, patient variability) remains unchanged.

### 2.2 The Decomposition in Bayesian Linear Regression (Recap)

We saw this concretely in the BLR notes. The predictive variance at a new input $\mathbf{x}_*$ is:

$$\sigma^2_{\text{pred}}(\mathbf{x}_*) = \underbrace{\beta^{-1}}_{\text{aleatoric}} + \underbrace{\boldsymbol{\phi}_*^T \mathbf{S}_N \boldsymbol{\phi}_*}_{\text{epistemic}}$$

The first term is observation noise — fixed, input-independent, and irreducible. The second is parameter uncertainty — it depends on $\mathbf{x}_*$ (larger where data is sparse) and shrinks as $N \to \infty$ because $\mathbf{S}_N \to \mathbf{0}$.

For linear regression, this decomposition is exact and closed-form. The rest of these notes is about recovering analogous decompositions for models where the posterior over parameters is not available in closed form.

### 2.3 Homoscedastic vs. Heteroscedastic Aleatoric Uncertainty

Not all aleatoric uncertainty is equal across the input space:

- **Homoscedastic:** The noise variance $\sigma^2$ is constant across all inputs. This is the assumption in standard linear regression. The BLR running example used $\sigma^2 = 0.1$ everywhere.

- **Heteroscedastic:** The noise variance $\sigma^2(\mathbf{x})$ varies with the input. This is the common case in practice — a house price model is more uncertain in luxury markets (high variance, few comparable sales) than in dense suburban markets (low variance, many comparables), even conditional on all observed features.

Standard models with a fixed loss function (MSE, cross-entropy) implicitly assume homoscedastic noise. Section 6 shows how to make the model predict its own noise level as a function of the input — the **heteroscedastic** formulation.

### 2.4 Why the Distinction Matters in Practice

| | Aleatoric | Epistemic |
|---|---|---|
| **Cause** | Inherent data noise, unobserved variables | Limited training data, model misspecification |
| **Reducible?** | No (for fixed features) | Yes (with more data) |
| **Input-dependence** | Can vary (heteroscedastic) or not | Always input-dependent — highest where training data is sparse |
| **Response** | Collect better features, accept irreducible noise floor | Collect more data, flag for human review, defer prediction |
| **BLR analogue** | $\beta^{-1}$ | $\boldsymbol{\phi}_*^T \mathbf{S}_N \boldsymbol{\phi}_*$ |

A model that conflates the two gives the same "uncertain" label to a genuinely ambiguous input (aleatoric — no action possible) and a rare input the model hasn't learned about (epistemic — more data would help). In safety-critical applications, this distinction drives decisions: epistemic uncertainty should trigger human review or data collection; aleatoric uncertainty should inform the user that the task is inherently noisy.

Aleatoric uncertainty is the noise floor you cannot reduce; epistemic uncertainty is the knowledge gap you can. In Bayesian linear regression, the two fall out as separate additive terms in the predictive variance. For neural networks, recovering this decomposition requires approximate methods — which is the subject of the next four sections.

---

## 3. Bayesian Neural Networks: The Intractable Ideal

The Bayesian treatment of neural networks follows the exact same recipe as BLR — prior, posterior, marginalization — but breaks down at each step because the likelihood is no longer quadratic in the weights. This section traces that analogy, identifies exactly where the closed-form solution fails, and introduces the two main approximation families that attempt to recover it: the Laplace approximation and variational inference. These are the theoretical foundation on which the practical methods in Sections 4 and 5 rest.

### 3.1 The Weight-Space View

A neural network with weights $\mathbf{w}$ defines a likelihood $p(\mathcal{D} \mid \mathbf{w})$ just as linear regression does. The Bayesian recipe is identical in principle:

$$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})}{p(\mathcal{D})}, \qquad p(y_* \mid \mathbf{x}_*, \mathcal{D}) = \int p(y_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w}$$

Place a prior $p(\mathbf{w})$ (typically $\mathcal{N}(\mathbf{0}, \alpha^{-1}I)$, equivalent to weight decay), observe data, update to the posterior, and marginalize for predictions.

**Where it breaks down:** In BLR, the likelihood was $\exp(-\frac{\beta}{2}\|\mathbf{y} - \boldsymbol{\Phi}\mathbf{w}\|^2)$ — a quadratic in **w**. Multiplied by a Gaussian prior, the exponent remained quadratic, and completing the square gave us a Gaussian posterior. In a neural network, the likelihood is $\exp(-\frac{\beta}{2}\|\mathbf{y} - f_{\mathbf{w}}(\mathbf{X})\|^2)$ where $f_{\mathbf{w}}$ is a nonlinear function of **w** (compositions of affine transforms and activations). The exponent is no longer quadratic in **w**, so:

1. The posterior $p(\mathbf{w} \mid \mathcal{D})$ is not Gaussian — it is a complex, multimodal distribution over millions of dimensions.
2. The marginal likelihood $p(\mathcal{D}) = \int p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})\, d\mathbf{w}$ is an intractable integral.
3. The predictive integral $\int p(y_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w}$ is also intractable.

All three integrals are over the full weight space — for a ResNet-50, that is $\sim 25$ million dimensions. No quadrature method can handle this. The solution is to approximate the posterior.

### 3.2 The Laplace Approximation

**Idea:** Find the MAP estimate $\hat{\mathbf{w}}_{\text{MAP}}$ (the mode of the posterior), then approximate the posterior as a Gaussian centred at the mode with covariance given by the inverse Hessian of the negative log-posterior.

The log-posterior is:

$$\log p(\mathbf{w} \mid \mathcal{D}) = \log p(\mathcal{D} \mid \mathbf{w}) + \log p(\mathbf{w}) - \log p(\mathcal{D})$$

Taking a second-order Taylor expansion around $\hat{\mathbf{w}}_{\text{MAP}}$:

$$\log p(\mathbf{w} \mid \mathcal{D}) \approx \log p(\hat{\mathbf{w}} \mid \mathcal{D}) + \underbrace{\nabla_{\mathbf{w}} \log p(\mathbf{w} \mid \mathcal{D})\big|_{\hat{\mathbf{w}}}^T}_{= \,\mathbf{0}} (\mathbf{w} - \hat{\mathbf{w}}) - \frac{1}{2}(\mathbf{w} - \hat{\mathbf{w}})^T \mathbf{H}\, (\mathbf{w} - \hat{\mathbf{w}})$$

The first-order term vanishes because $\hat{\mathbf{w}}_{\text{MAP}}$ is a stationary point of the log-posterior — the gradient is zero at the mode by definition of MAP. This leaves:

$$\log p(\mathbf{w} \mid \mathcal{D}) \approx \log p(\hat{\mathbf{w}} \mid \mathcal{D}) - \frac{1}{2}(\mathbf{w} - \hat{\mathbf{w}})^T \mathbf{H}\, (\mathbf{w} - \hat{\mathbf{w}})$$

where $\mathbf{H} = -\nabla^2_{\mathbf{w}} \log p(\mathbf{w} \mid \mathcal{D})\big|_{\mathbf{w} = \hat{\mathbf{w}}}$ is the Hessian of the negative log-posterior evaluated at the mode. Exponentiating both sides gives a Gaussian:

$$p(\mathbf{w} \mid \mathcal{D}) \approx \mathcal{N}(\hat{\mathbf{w}}_{\text{MAP}},\, \mathbf{H}^{-1})$$

**Connection to BLR:** For linear regression, the log-posterior is exactly quadratic — no approximation is needed. The Hessian is $\mathbf{S}_N^{-1} = \alpha I + \beta \boldsymbol{\Phi}^T\boldsymbol{\Phi}$ (constant in **w**), and $\mathbf{H}^{-1} = \mathbf{S}_N$. The Laplace approximation recovers the exact BLR posterior.

**The problem for neural networks:** $\mathbf{H}$ is a $D \times D$ matrix where $D$ is the number of parameters. For a network with 25 million weights, storing $\mathbf{H}$ requires $\sim 2.3$ petabytes. Even forming it is out of the question.

**Practical workarounds:**

- **Diagonal approximation:** Assume $\mathbf{H}$ is diagonal — each weight's uncertainty is independent. This reduces storage to $O(D)$ but ignores correlations between weights. Overly optimistic in some directions, overly pessimistic in others.
- **KFAC (Kronecker-Factored Approximate Curvature):** Approximate the Hessian as a Kronecker product of smaller matrices, one per layer. Captures within-layer correlations at manageable cost. Introduced by Martens & Grosse (2015); applied to Laplace approximation by Ritter et al. (2018).
- **Last-layer Laplace:** Apply the Laplace approximation only to the final linear layer, treating earlier layers as a fixed feature extractor. The last layer is linear in its weights, so the Hessian for that layer is exact — we are back to BLR on learned features. This is cheap, easy to implement, and often surprisingly effective (Daxberger et al. 2021).

### 3.3 Variational Inference (VI)

**Idea:** Rather than approximating the posterior locally (as Laplace does at the mode), choose a tractable family of distributions $q_\theta(\mathbf{w})$ and find the member closest to the true posterior.

"Closest" is measured by the KL divergence from $q$ to the posterior:

$$\text{KL}(q_\theta(\mathbf{w}) \| p(\mathbf{w} \mid \mathcal{D})) = \int q_\theta(\mathbf{w}) \log \frac{q_\theta(\mathbf{w})}{p(\mathbf{w} \mid \mathcal{D})}\, d\mathbf{w}$$

This cannot be computed directly (it requires $p(\mathbf{w} \mid \mathcal{D})$, which is what we don't have). But it can be rearranged into a computable objective. The key move is to substitute Bayes' rule for the intractable posterior inside the log. Starting from the definition:

$$\text{KL}(q \| p(\mathbf{w} \mid \mathcal{D})) = \int q_\theta(\mathbf{w}) \log \frac{q_\theta(\mathbf{w})}{p(\mathbf{w} \mid \mathcal{D})}\, d\mathbf{w}$$

Replace $p(\mathbf{w} \mid \mathcal{D})$ with $\frac{p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})}{p(\mathcal{D})}$ (Bayes' rule):

$$= \int q_\theta(\mathbf{w}) \log \frac{q_\theta(\mathbf{w}) \cdot p(\mathcal{D})}{p(\mathcal{D} \mid \mathbf{w})\, p(\mathbf{w})}\, d\mathbf{w}$$

Split the log of the product into a sum of logs:

$$= \int q_\theta(\mathbf{w}) \left[\log \frac{q_\theta(\mathbf{w})}{p(\mathbf{w})} - \log p(\mathcal{D} \mid \mathbf{w}) + \log p(\mathcal{D})\right] d\mathbf{w}$$

The third term $\log p(\mathcal{D})$ is constant in $\mathbf{w}$, so it factors out of the integral (and $\int q_\theta\, d\mathbf{w} = 1$). The first term is the KL divergence from $q_\theta$ to the prior. The second is the expected log-likelihood under $q_\theta$:

$$= \text{KL}(q_\theta(\mathbf{w}) \| p(\mathbf{w})) - \mathbb{E}_{q_\theta}[\log p(\mathcal{D} \mid \mathbf{w})] + \log p(\mathcal{D})$$

Rearranging:

$$\text{KL}(q \| p(\mathbf{w} \mid \mathcal{D})) = -\underbrace{\left[\mathbb{E}_{q}[\log p(\mathcal{D} \mid \mathbf{w})] - \text{KL}(q_\theta(\mathbf{w}) \| p(\mathbf{w}))\right]}_{\text{ELBO}(\theta)} + \log p(\mathcal{D})$$

Since $\log p(\mathcal{D})$ is a constant (it does not depend on $\theta$), minimizing $\text{KL}(q \| p(\mathbf{w} \mid \mathcal{D}))$ is equivalent to maximizing the **Evidence Lower Bound (ELBO)**:

$$\text{ELBO}(\theta) = \underbrace{\mathbb{E}_{q_\theta}[\log p(\mathcal{D} \mid \mathbf{w})]}_{\text{expected log-likelihood}} - \underbrace{\text{KL}(q_\theta(\mathbf{w}) \| p(\mathbf{w}))}_{\text{complexity penalty}}$$

The name "evidence lower bound" comes from rearranging the equation above: $\log p(\mathcal{D}) = \text{ELBO}(\theta) + \text{KL}(q \| p(\mathbf{w} \mid \mathcal{D}))$. Since KL divergence is non-negative, $\text{ELBO}(\theta) \leq \log p(\mathcal{D})$ — the ELBO is always a lower bound on the log model evidence, with equality when $q_\theta = p(\mathbf{w} \mid \mathcal{D})$.

**How to read the ELBO:** The first term rewards variational distributions that place mass on weights that explain the data well. The second term penalizes distributions that stray far from the prior — the more the approximate posterior deviates from the prior, the higher the complexity cost. This is a continuous, distributional generalization of MAP's log-likelihood + log-prior objective: MAP optimizes a single point **w**; VI optimizes an entire distribution $q_\theta$.

**Bayes by Backprop (Blundell et al. 2015).** The most straightforward instantiation for neural networks. Let $q_\theta(\mathbf{w}) = \prod_i \mathcal{N}(w_i;\, \mu_i, \sigma_i^2)$ — a fully factorized Gaussian (mean-field approximation). Each weight has its own learned mean $\mu_i$ and standard deviation $\sigma_i$, doubling the number of parameters. Training uses the **reparameterization trick**: sample $\epsilon_i \sim \mathcal{N}(0,1)$, set $w_i = \mu_i + \sigma_i \epsilon_i$, and backpropagate through $\mu_i$ and $\sigma_i$.

At test time, draw $T$ weight samples from $q_\theta$, run a forward pass for each, and average:

$$p(y_* \mid \mathbf{x}_*, \mathcal{D}) \approx \frac{1}{T}\sum_{t=1}^T p(y_* \mid \mathbf{x}_*, \mathbf{w}^{(t)}), \qquad \mathbf{w}^{(t)} \sim q_\theta(\mathbf{w})$$

**Limitations of VI:**

- **Mean-field underestimates uncertainty.** The fully factorized assumption ignores all weight correlations, producing posteriors that are too narrow. The approximate posterior tends to "snap" to one mode of the true posterior, missing others entirely — a consequence of minimizing $\text{KL}(q \| p)$ rather than $\text{KL}(p \| q)$. The reverse KL ($\text{KL}(q \| p)$, the direction VI uses) is mode-seeking: $q$ pays a heavy penalty for placing mass where $p$ is small, so it collapses onto a single mode rather than spanning all of them. The forward KL ($\text{KL}(p \| q)$) would be mode-covering — it penalizes $q$ for missing any region where $p$ has mass — but is intractable to optimize because it requires expectations under the true posterior $p$.

[FIG:ORIGINAL — Side-by-side comparison of mode-seeking (reverse KL) and mode-covering (forward KL) behavior on a bimodal target distribution p(w): left panel shows q optimized under KL(q||p) collapsing onto a single mode of p; right panel shows q optimized under KL(p||q) spreading to cover both modes, resulting in a wider but less concentrated approximation.]
- **Doubling parameters.** Storing $(\mu_i, \sigma_i)$ for every weight doubles memory. For large models this is non-trivial.
- **Training instability.** The gradient estimator for the ELBO has high variance, often requiring careful learning rate schedules and warm-up.

### 3.4 The Practical Landscape

There is a third major approximation family not yet discussed: **Markov Chain Monte Carlo (MCMC)**. MCMC constructs a Markov chain whose stationary distribution is the posterior, then collects samples from that chain to approximate posterior expectations. Unlike the Laplace approximation (which commits to a single Gaussian) or VI (which optimizes within a restricted family), MCMC is *asymptotically exact* — given enough samples, it converges to the true posterior regardless of its shape. However, MCMC has not gained traction for deep learning because the cost per sample scales with a full gradient computation over the dataset, mixing is slow in the high-dimensional, multimodal weight spaces of neural networks, and diagnosing convergence is difficult. Stochastic gradient MCMC variants (e.g., SGLD — Welling & Teh 2011) reduce per-step cost by using mini-batch gradients, but in practice they still lag behind simpler methods in the accuracy–cost trade-off for large models.

Neither the Laplace approximation, standard VI, nor MCMC has become the default uncertainty method for large-scale deep learning. The Laplace approximation requires Hessian computation (or approximation); VI doubles parameters and introduces training complexity; MCMC is expensive and hard to tune. In practice, two methods dominate due to their simplicity: **MC Dropout** (Section 4) and **Deep Ensembles** (Section 5). Both can be understood as approximate Bayesian inference — but their appeal is that they require minimal changes to standard training pipelines.

The Bayesian treatment of neural networks is conceptually identical to BLR — place a prior, compute the posterior, marginalize — but the nonlinearity of the network makes all three integrals intractable. The three main approximation strategies are: the Laplace approximation (Gaussian at the posterior mode), variational inference (optimize a tractable approximate posterior via the ELBO), and MCMC (asymptotically exact sampling, but expensive). All three have significant practical limitations for large models. The methods that have gained the widest adoption — MC Dropout and Deep Ensembles — succeed largely because they piggyback on standard training infrastructure. We develop them next.

---

## 4. MC Dropout: Dropout as Approximate Bayesian Inference

MC Dropout (Gal & Ghahramani 2016) reinterprets the standard dropout regularization technique as approximate variational inference — turning a trick that was already part of most training pipelines into an uncertainty estimation method at near-zero cost. This section explains the theoretical argument, derives the predictive mean and variance, shows how to decompose uncertainty into aleatoric and epistemic components, and discusses the method's strengths and significant limitations.

### 4.1 Standard Dropout (Recap)

Dropout (Srivastava et al. 2014) is a regularization technique: during training, each hidden unit is independently set to zero with probability $p$ (the *dropout rate*). This prevents co-adaptation of neurons and acts as an implicit ensemble over $2^H$ subnetworks (where $H$ is the number of hidden units).

At test time, standard practice is to use all units but scale their activations by $(1 - p)$ to match the expected activation during training. This produces a single deterministic forward pass — no randomness, no uncertainty.

### 4.2 The Key Insight: Keep Dropout On at Test Time

Gal & Ghahramani (2016) showed that **applying dropout at test time and running multiple stochastic forward passes** is mathematically equivalent to approximate variational inference with a specific variational family.

The variational distribution is: for each weight matrix $\mathbf{W}_l$ in layer $l$, draw a random diagonal mask $\text{diag}(\mathbf{z}_l)$ where each $z_{l,j} \sim \text{Bernoulli}(1 - p)$, and use $\widetilde{\mathbf{W}}_l = \mathbf{W}_l \cdot \text{diag}(\mathbf{z}_l)$. The variational parameters are the weight matrices $\{\mathbf{W}_l\}$ themselves (learned during training); the stochasticity comes from the Bernoulli masks.

Gal & Ghahramani showed that training with dropout (minimizing cross-entropy or MSE + $\ell_2$ regularization) is equivalent to maximizing a lower bound on the ELBO for this variational family. Recall that the ELBO has two terms: the expected log-likelihood (reward data fit) and the KL penalty (keep the approximate posterior close to the prior). Standard dropout training approximately optimizes both: the MSE or cross-entropy loss serves as the expected log-likelihood term, while the $\ell_2$ regularization term acts as the KL divergence from the variational posterior to a Gaussian prior — $\ell_2$ regularization $\lambda\|\mathbf{w}\|^2$ is the negative log of a Gaussian prior $\mathcal{N}(\mathbf{0}, \lambda^{-1}I)$ up to a constant, so the regularization term does double duty as the KL penalty that keeps $q_\theta$ close to this prior (the same connection between weight decay and Gaussian priors derived in the MAP notes). The Bernoulli masks contribute additional entropy terms that complete the bound. The full proof (Gal & Ghahramani 2016, Appendix 1) involves showing that these components together lower-bound the ELBO for the specific Bernoulli variational family described above; we take this result on authority here as the derivation requires several pages of careful bookkeeping over the layer-wise mask distributions.

### 4.3 Predictive Mean and Variance

At test time, run $T$ stochastic forward passes with dropout active, producing predictions $\{\hat{y}_*^{(t)}\}_{t=1}^T$:

$$\hat{y}_*^{(t)} = f_{\widetilde{\mathbf{w}}^{(t)}}(\mathbf{x}_*), \qquad \widetilde{\mathbf{w}}^{(t)} \sim q_\theta(\mathbf{w})$$

The Monte Carlo estimates of the predictive moments are:

**Predictive mean** (the "Bayesian model average"):

$$\mathbb{E}[y_* \mid \mathbf{x}_*, \mathcal{D}] \approx \frac{1}{T}\sum_{t=1}^T \hat{y}_*^{(t)}$$

**Predictive variance** (for regression):

$$\text{Var}[y_* \mid \mathbf{x}_*, \mathcal{D}] \approx \underbrace{\sigma^2_{\text{noise}}}_{\text{aleatoric}} + \underbrace{\frac{1}{T}\sum_{t=1}^T \left(\hat{y}_*^{(t)}\right)^2 - \left(\frac{1}{T}\sum_{t=1}^T \hat{y}_*^{(t)}\right)^2}_{\text{epistemic (spread across dropout samples)}}$$

**Where does $\sigma^2_{\text{noise}}$ come from?** In a standard MC Dropout setup where the network outputs a single point prediction $\hat{y}$, $\sigma^2_{\text{noise}}$ is a fixed scalar that must be specified or estimated externally — for instance, from the average squared residual on a validation set: $\hat{\sigma}^2 = \frac{1}{N_\text{val}}\sum_i (y_i - \hat{y}_i)^2$. This is a *homoscedastic* plug-in estimate — it does not vary with the input. To get input-dependent aleatoric uncertainty, the network must output its own variance estimate via a heteroscedastic head (Section 6). The MC Dropout variance decomposition as written is therefore incomplete without Section 6: it captures epistemic uncertainty directly from the forward-pass spread, but the aleatoric term is either a global constant or requires the architecture extension described later.

The epistemic term is the sample variance of the $T$ forward-pass outputs. When all forward passes agree, the model is confident (low epistemic uncertainty). When they disagree, the input lies in a region of weight space that is poorly constrained — different subnetworks make different predictions.

**For classification**, replace variance with predictive entropy. Given $T$ stochastic softmax outputs $\hat{\mathbf{p}}^{(t)} = \text{softmax}(f_{\widetilde{\mathbf{w}}^{(t)}}(\mathbf{x}_*))$:

$$\bar{\mathbf{p}} = \frac{1}{T}\sum_{t=1}^T \hat{\mathbf{p}}^{(t)}$$

**Predictive entropy** (total uncertainty):

$$\mathbb{H}[\bar{\mathbf{p}}] = -\sum_c \bar{p}_c \log \bar{p}_c$$

**Mutual information** (epistemic uncertainty, often called BALD — Bayesian Active Learning by Disagreement):

$$\mathbb{I}[y_*; \mathbf{w} \mid \mathbf{x}_*, \mathcal{D}] = \mathbb{H}[\bar{\mathbf{p}}] - \frac{1}{T}\sum_{t=1}^T \mathbb{H}[\hat{\mathbf{p}}^{(t)}]$$

Mutual information $\mathbb{I}[y_*; \mathbf{w} \mid \mathbf{x}_*, \mathcal{D}]$ measures how much knowing the true weights $\mathbf{w}$ would reduce our uncertainty about the prediction $y_*$ — which is precisely the definition of epistemic uncertainty (uncertainty due to not knowing $\mathbf{w}$). If $\mathbf{w}$ were known, only aleatoric uncertainty would remain, so the reduction is entirely epistemic.

Operationally, this is total entropy minus the average entropy of individual forward passes. It is high when the *average* prediction is uncertain but individual forward passes are each confident in *different* classes — the hallmark of epistemic uncertainty. It is low when every forward pass is uncertain in the same way — aleatoric uncertainty.

**A concrete example makes this crisp.** Consider binary classification ($C = 2$) with $T = 3$ forward passes.

*Case 1 — High epistemic uncertainty (disagreement):* Three forward passes produce $\hat{\mathbf{p}}^{(1)} = [0.95, 0.05]$, $\hat{\mathbf{p}}^{(2)} = [0.05, 0.95]$, $\hat{\mathbf{p}}^{(3)} = [0.50, 0.50]$. The average is $\bar{\mathbf{p}} = [0.50, 0.50]$, so $\mathbb{H}[\bar{\mathbf{p}}] = -2 \times 0.5 \ln 0.5 = 0.693$ nats. Each individual pass has low entropy: $\mathbb{H}[\hat{\mathbf{p}}^{(1)}] = \mathbb{H}[\hat{\mathbf{p}}^{(2)}] = 0.199$, $\mathbb{H}[\hat{\mathbf{p}}^{(3)}] = 0.693$. The average individual entropy is $\frac{1}{3}(0.199 + 0.199 + 0.693) = 0.364$. Mutual information: $0.693 - 0.364 = 0.329$ nats — substantial. The model is *collectively* uncertain because different subnetworks disagree, not because any single one is confused. This is the signature of epistemic uncertainty.

*Case 2 — High aleatoric uncertainty (unanimous confusion):* Three forward passes all produce $\hat{\mathbf{p}}^{(t)} = [0.50, 0.50]$. The average is $\bar{\mathbf{p}} = [0.50, 0.50]$, so $\mathbb{H}[\bar{\mathbf{p}}] = 0.693$ — identical total entropy as Case 1. But every individual pass also has entropy $0.693$, so the average individual entropy is $0.693$. Mutual information: $0.693 - 0.693 = 0$ nats. No epistemic uncertainty at all — the model is uncertain because the input is genuinely ambiguous, not because different subnetworks disagree. All the uncertainty is aleatoric.

### 4.4 Implementation

MC Dropout is appealing because it requires almost no code changes:

```python
import torch
import torch.nn as nn

class MCDropoutModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(p=drop_p)

    def forward(self, x):
        x = self.drop(torch.relu(self.fc1(x)))
        x = self.drop(torch.relu(self.fc2(x)))
        return self.fc3(x)

def predict_with_uncertainty(model, x, T=50):
    model.train()  # keeps dropout active
    preds = torch.stack([model(x) for _ in range(T)])       # (T, batch, output_dim)
    pred_mean = preds.mean(dim=0)
    epistemic_var = preds.var(dim=0)                         # spread across dropout masks
    return pred_mean, epistemic_var
```

The only non-standard step is `model.train()` at test time — this keeps the dropout layers active instead of disabling them as `model.eval()` would.

### 4.5 Strengths and Limitations

**Strengths:**
- **Near-zero implementation cost.** Any model already trained with dropout can produce uncertainty estimates by running $T$ forward passes at test time.
- **Theoretically grounded.** The connection to variational inference gives the uncertainty estimates a principled interpretation.
- **No retraining required.** Existing checkpoints can be used directly.

**Limitations:**
- **$T$-fold inference cost.** Each prediction requires $T$ forward passes (typically $T = 10$–$100$). For latency-sensitive applications, this is a real cost.
- **Dropout rate sensitivity.** The uncertainty quality depends heavily on the dropout rate $p$, which was typically tuned for regularization performance, not uncertainty quality. These are different objectives. Gal & Ghahramani suggest tuning $p$ via a grid search over validation log-likelihood, but this is rarely done in practice.
- **Underestimates uncertainty.** The Bernoulli variational family is restrictive. Like all mean-field VI, it tends to underestimate posterior variance. Empirically, MC Dropout uncertainty is often less well-calibrated than Deep Ensembles (Lakshminarayanan et al. 2017, Ovadia et al. 2019).
- **Architecture dependence.** The theoretical guarantee requires dropout after *every* weight layer. Many modern architectures (ResNets, Transformers) use dropout sparingly or not at all, and adding dropout everywhere can degrade accuracy.

MC Dropout turns a standard regularization trick into an uncertainty estimation method by keeping dropout active at test time and interpreting the variation across stochastic forward passes as epistemic uncertainty. It is theoretically motivated as variational inference, trivial to implement, and requires no retraining — but it is $T\times$ slower at inference, sensitive to the dropout rate, and empirically tends to underestimate uncertainty compared to ensembles.

---

## 5. Deep Ensembles

Deep Ensembles (Lakshminarayanan et al. 2017) are currently the most empirically reliable method for uncertainty estimation in deep learning — and among the simplest. This section explains the method, connects it to Bayesian model averaging, derives the uncertainty decomposition via the law of total variance, and discusses why ensembles work as well as they do despite having no explicit probabilistic motivation.

### 5.1 The Method

Train $M$ independent neural networks (typically $M = 5$–$10$) from scratch, each with:
- **Different random initialization.** Xavier, Kaiming, or other standard initializers — but with different seeds.
- **Different data ordering.** Stochastic gradient descent sees mini-batches in a different random order for each ensemble member.
- (Optionally) **Different data subsets.** Bagging (bootstrap aggregation) can be used, though Lakshminarayanan et al. found it unnecessary.

Each network $f_{\mathbf{w}_m}$ is trained normally (same architecture, same loss, same hyperparameters). At test time, aggregate:

**Predictive mean:**

$$\bar{\mu}(\mathbf{x}_*) = \frac{1}{M}\sum_{m=1}^M \hat{y}_m(\mathbf{x}_*)$$

where $\hat{y}_m(\mathbf{x}_*)$ is the point prediction from ensemble member $m$.

**Predictive variance** (for regression): In the simplest case — where each member outputs only a point prediction $\hat{y}_m$ and aleatoric noise is treated as a fixed global estimate $\sigma^2_{\text{noise}}$ (just as in MC Dropout, Section 4.3) — the predictive variance is:

$$\sigma^2_{\text{pred}}(\mathbf{x}_*) = \sigma^2_{\text{noise}} + \underbrace{\frac{1}{M}\sum_{m=1}^M \left(\hat{y}_m(\mathbf{x}_*) - \bar{\mu}(\mathbf{x}_*)\right)^2}_{\text{epistemic variance (ensemble disagreement)}}$$

The epistemic term is the sample variance of the $M$ members' predictions — identical in spirit to the MC Dropout variance from Section 4.3, but with each sample coming from a *different trained model* rather than a different dropout mask.

When each ensemble member instead outputs both a mean $\mu_m$ and a learned variance $\sigma_m^2$ via a heteroscedastic output head (introduced in Section 6), the decomposition becomes richer. The **law of total variance** gives us:

$$\sigma^2_{\text{pred}}(\mathbf{x}_*) = \underbrace{\frac{1}{M}\sum_{m=1}^M \sigma_m^2(\mathbf{x}_*)}_{\text{mean aleatoric variance}} + \underbrace{\frac{1}{M}\sum_{m=1}^M \left(\mu_m(\mathbf{x}_*) - \bar{\mu}(\mathbf{x}_*)\right)^2}_{\text{epistemic variance (disagreement)}}$$

This is the full aleatoric + epistemic decomposition — the neural network analogue of the BLR result $\sigma^2_{\text{pred}} = \beta^{-1} + \boldsymbol{\phi}_*^T \mathbf{S}_N \boldsymbol{\phi}_*$, with input-dependent aleatoric uncertainty. We defer the heteroscedastic output head to Section 6 and return to this full formula there; for now, the key result is the epistemic term — ensemble disagreement.

**Deriving the law of total variance.** The decomposition above is not an approximation — it follows from a general identity. Let $M$ denote the ensemble member (a discrete random variable, uniform over $\{1, \ldots, M\}$), and $Y$ the prediction. Starting from the definition of variance and applying the tower property ($\mathbb{E}[Y] = \mathbb{E}_M[\mathbb{E}[Y \mid M]]$):

$$\text{Var}[Y] = \mathbb{E}[Y^2] - (\mathbb{E}[Y])^2$$

Add and subtract $\mathbb{E}_M[(\mathbb{E}[Y \mid M])^2]$:

$$= \underbrace{\mathbb{E}_M[\mathbb{E}[Y^2 \mid M]] - \mathbb{E}_M[(\mathbb{E}[Y \mid M])^2]}_{\mathbb{E}_M[\text{Var}[Y \mid M]]} + \underbrace{\mathbb{E}_M[(\mathbb{E}[Y \mid M])^2] - (\mathbb{E}_M[\mathbb{E}[Y \mid M]])^2}_{\text{Var}_M[\mathbb{E}[Y \mid M]]}$$

The first group is $\mathbb{E}_M[\text{Var}[Y \mid M]]$ (mean within-member variance — aleatoric). The second is $\text{Var}_M[\mathbb{E}[Y \mid M]]$ (variance of member means — epistemic). In summary:

$$\text{Var}[Y] = \underbrace{\mathbb{E}_M[\text{Var}[Y \mid M]]}_{\text{mean within-member variance}} + \underbrace{\text{Var}_M[\mathbb{E}[Y \mid M]]}_{\text{variance of member means}}$$

When ensemble members output only point predictions (no learned variance), the first term reduces to the plug-in noise estimate $\sigma^2_{\text{noise}}$. When they output $(\mu_m, \sigma_m^2)$ via a heteroscedastic head, the first term becomes $\frac{1}{M}\sum_m \sigma_m^2(\mathbf{x}_*)$ — input-dependent aleatoric uncertainty.

**For classification**, average the softmax outputs:

$$\bar{\mathbf{p}}(\mathbf{x}_*) = \frac{1}{M}\sum_{m=1}^M \text{softmax}(f_{\mathbf{w}_m}(\mathbf{x}_*))$$

Predictive entropy and mutual information are computed exactly as in Section 4.3, replacing dropout samples with ensemble members.

### 5.2 Why Ensembles Capture Epistemic Uncertainty

The loss landscape of a neural network has many local minima (or, more precisely, many regions of low loss separated by high-loss barriers). Different random initializations lead SGD to converge to different solutions — different "modes" of the loss landscape. Each mode represents a different function that fits the training data well.

Near the training data, all modes agree — they all fit the observed points. Away from the training data, they diverge — each mode extrapolates differently based on the particular basin of the loss landscape it settled into. This disagreement among ensemble members is a direct signal of epistemic uncertainty.

[FIG:ORIGINAL — Five neural networks (ensemble members) with different random initializations all fit the same 1D training data (black dots). Within the training range the curves nearly overlap; outside it each network extrapolates differently. The fanning of curves in the extrapolation region is a direct visual signature of epistemic uncertainty — ensemble disagreement.]

The connection to Bayesian inference is approximate but illuminating. The true Bayesian predictive distribution averages over the full posterior $p(\mathbf{w} \mid \mathcal{D})$:

$$p(y_* \mid \mathbf{x}_*, \mathcal{D}) = \int p(y_* \mid \mathbf{x}_*, \mathbf{w})\, p(\mathbf{w} \mid \mathcal{D})\, d\mathbf{w}$$

Deep Ensembles approximate this with a mixture of $M$ point masses at the converged weights:

$$p(y_* \mid \mathbf{x}_*, \mathcal{D}) \approx \frac{1}{M}\sum_{m=1}^M p(y_* \mid \mathbf{x}_*, \mathbf{w}_m)$$

This is a crude approximation of the integral — $M = 5$ point masses in a space of millions of dimensions. Yet it works remarkably well, consistently outperforming more sophisticated Bayesian methods in large-scale benchmarks (Ovadia et al. 2019, Ashukha et al. 2020). The reason appears to be that the $M$ ensemble members land in *different modes* of the posterior, capturing the dominant source of predictive diversity — inter-mode variation — even if they miss intra-mode variation entirely.

### 5.3 Comparison with MC Dropout

| | MC Dropout | Deep Ensembles |
|---|---|---|
| **Training cost** | $1\times$ (train once) | $M\times$ (train $M$ models) |
| **Inference cost** | $T$ forward passes, 1 model | $M$ forward passes, $M$ models |
| **Storage** | 1 model | $M$ models |
| **Diversity source** | Random dropout masks (same weights) | Different converged weights (different modes) |
| **Calibration quality** | Moderate | Best in class (empirically) |
| **Theoretical grounding** | Variational inference (specific family) | Bayesian model averaging (approximate) |

The key empirical finding is that **inter-mode diversity (ensembles) captures more uncertainty than intra-mode perturbation (dropout)**. Different initializations find genuinely different functions; different dropout masks explore the neighbourhood of a single function.

### 5.4 Practical Considerations

- **$M = 5$ is usually sufficient.** Lakshminarayanan et al. found diminishing returns beyond $M = 5$ for both calibration and accuracy. This makes ensembles $5\times$ the cost of a single model — significant, but manageable for many applications.
- **Ensembles also improve accuracy.** The ensemble mean is a better predictor than any single member, due to the classic variance-reduction effect of averaging. This means the cost of ensembling buys two things simultaneously: better point predictions and uncertainty estimates.
- **Snapshot ensembles and other cost-reduction tricks.** Training $M$ full models is expensive. Alternatives include: cyclical learning rate schedules that visit multiple modes during a single training run (Huang et al. 2017), late-stage branching from a single pre-trained trunk, or hyperparameter ensembles (same architecture, different learning rates). These reduce cost at the expense of diversity.

Deep Ensembles train $M$ networks from different random initializations and aggregate their predictions. The spread among ensemble members — their disagreement — directly measures epistemic uncertainty. Despite minimal theoretical sophistication, ensembles produce the best-calibrated uncertainty estimates in practice, because different initializations explore different modes of the loss landscape and those modes diverge where training data is sparse. The main cost is $M\times$ training and storage.

---

## 6. Heteroscedastic Models: Learning Input-Dependent Noise

Sections 3–5 focused on epistemic uncertainty — uncertainty about the model's parameters. Now we turn to the other half: aleatoric uncertainty that varies across the input space. The key idea is to make the model output not just a prediction but also its own noise estimate, trained end-to-end using a modified loss function.

### 6.1 Motivation: Why Fixed Noise Is Often Wrong

Standard regression training minimizes MSE:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$

This implicitly assumes homoscedastic noise — the same $\sigma^2$ everywhere. But many real datasets have input-dependent noise:

- **Finance:** Volatility varies across asset classes, time periods, and market regimes.
- **Sensor data:** Measurement noise depends on environmental conditions (temperature, humidity, distance).
- **Medical imaging:** Image quality varies by scanner, patient motion, and tissue type.

A homoscedastic model that encounters a high-noise region will either overfit to the noise (if flexible enough) or underfit the signal (if not). What we want is a model that says: "I predict $\hat{y} = 4.2$ with high confidence here, but $\hat{y} = 3.8$ with low confidence there — not because I haven't seen enough data, but because the target is inherently noisier at that input."

### 6.2 The Gaussian Negative Log-Likelihood Loss

Instead of predicting a single scalar $\hat{y}$, the network outputs two quantities:

$$f_{\mathbf{w}}(\mathbf{x}) = (\mu(\mathbf{x}),\; \sigma^2(\mathbf{x}))$$

The predictive distribution for a single observation is $p(y \mid \mathbf{x}, \mathbf{w}) = \mathcal{N}(y;\, \mu(\mathbf{x}),\, \sigma^2(\mathbf{x}))$. Training maximizes the likelihood — or equivalently minimizes the **negative log-likelihood (NLL)**:

$$\mathcal{L}_{\text{NLL}} = \frac{1}{N}\sum_{i=1}^N \left[\frac{(y_i - \mu(\mathbf{x}_i))^2}{2\sigma^2(\mathbf{x}_i)} + \frac{1}{2}\log \sigma^2(\mathbf{x}_i)\right]$$

**How to read this loss:** The first term is the squared residual *weighted by the predicted variance* — large residuals are penalized less where the model predicts high noise. This might seem like the model could cheat by predicting $\sigma^2 \to \infty$ everywhere (every residual gets divided by infinity). The second term, $\frac{1}{2}\log \sigma^2$, prevents this: it penalizes large predicted variances directly. The balance between the two terms forces the model to predict high $\sigma^2$ only where the residuals genuinely warrant it.

**Implementation detail:** The network typically outputs $\log \sigma^2(\mathbf{x})$ rather than $\sigma^2(\mathbf{x})$ directly, to ensure positivity without constrained optimization. A common architecture uses a shared trunk with two output heads:

```python
import torch
import torch.nn as nn

class HeteroscedasticNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.trunk(x)
        mu = self.mean_head(h)
        log_var = self.logvar_head(h)
        return mu, log_var

def gaussian_nll_loss(y, mu, log_var):
    var = torch.exp(log_var)
    return 0.5 * (((y - mu)**2) / var + log_var).mean()
```

### 6.3 Combining Aleatoric and Epistemic Uncertainty

The full uncertainty picture emerges by combining a heteroscedastic output head with an ensemble (or MC Dropout). Each ensemble member $m$ outputs $(\mu_m(\mathbf{x}_*), \sigma_m^2(\mathbf{x}_*))$. The total predictive variance is:

$$\sigma^2_{\text{total}}(\mathbf{x}_*) = \underbrace{\frac{1}{M}\sum_{m=1}^M \sigma_m^2(\mathbf{x}_*)}_{\text{aleatoric (mean predicted noise)}} + \underbrace{\frac{1}{M}\sum_{m=1}^M \left(\mu_m(\mathbf{x}_*) - \bar{\mu}(\mathbf{x}_*)\right)^2}_{\text{epistemic (ensemble disagreement)}}$$

This gives us the same clean decomposition that fell out analytically in BLR — aleatoric + epistemic — but now for arbitrarily complex models. The aleatoric term is input-dependent (heteroscedastic), and the epistemic term captures model uncertainty.

**This is the practical payoff of the entire framework:** we can now look at a prediction and say not just "the model is uncertain" but *why* — because the input is inherently noisy (collect better features) or because the model hasn't seen enough similar inputs (collect more data, or defer to a human).

### 6.4 A Subtlety: The Learned Variance Is Not Calibrated by Default

The Gaussian NLL is a strictly proper scoring rule (Section 8.4), so in the infinite-data limit with a correctly specified model, the learned variance would be perfectly calibrated. In practice, finite training data and model misspecification — the network's functional form cannot exactly represent the true conditional distribution — mean the optimum found during training does not match the population optimum. The result: the NLL loss encourages the model to match the *relative* noise levels across the input space — it will learn that region A is noisier than region B — but the *absolute* scale of the predicted variance depends on the model's functional form and can be systematically too small or too large.

**How to diagnose this:** Compute the fraction of held-out targets that fall within the predicted $\alpha$-level confidence interval for several values of $\alpha$ (e.g., 50%, 80%, 90%, 95%). If the model predicts $\mathcal{N}(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))$, the $\alpha$-level interval is $\mu \pm z_{\alpha/2}\, \sigma$. If the empirical coverage is consistently below the nominal level (e.g., only 82% of targets fall in the "90%" interval), the model's variances are too small.

**How to fix it:** The simplest correction is **variance scaling** — the regression analogue of temperature scaling. Learn a single scalar $s > 0$ on a held-out calibration set by minimizing the Gaussian NLL with rescaled variance:

$$s^* = \arg\min_s \; \frac{1}{N_\text{cal}} \sum_{i=1}^{N_\text{cal}} \left[\frac{(y_i - \mu(\mathbf{x}_i))^2}{2\, s \cdot \sigma^2(\mathbf{x}_i)} + \frac{1}{2}\log(s \cdot \sigma^2(\mathbf{x}_i))\right]$$

At test time, replace $\sigma^2(\mathbf{x})$ with $s^* \cdot \sigma^2(\mathbf{x})$. This preserves the relative ordering of variances while correcting the global scale. Just as temperature scaling fixes classification calibration with one parameter, variance scaling fixes regression calibration with one parameter.

Making the network output both a mean and a variance, trained with the Gaussian NLL loss, allows the model to learn input-dependent aleatoric uncertainty. Combining this with an epistemic method (ensembles or MC Dropout) recovers the full aleatoric + epistemic decomposition for any model architecture. The NLL loss balances two forces: fitting residuals and avoiding inflated variance predictions. The result is a model that can distinguish "I'm uncertain because this region is inherently noisy" from "I'm uncertain because I haven't seen enough data like this."

---

## 7. Calibration: Do Predicted Probabilities Mean What They Say?

Even with well-designed uncertainty methods, predicted probabilities may not match reality — a model that says "80% chance of rain" is only useful if it actually rains about 80% of the time the model says this. This section defines calibration precisely, introduces the tools to measure it (reliability diagrams, Expected Calibration Error), explains why modern neural networks are systematically miscalibrated, and presents the dominant post-hoc fix: temperature scaling.

### 7.1 What Calibration Means

A probabilistic predictor is **perfectly calibrated** if:

$$P(Y = c \mid \hat{p}(Y = c \mid X) = q) = q \qquad \text{for all classes } c \text{ and all } q \in [0, 1]$$

In words: among all inputs where the model predicts "class $c$ with probability $q$," the true fraction that are actually class $c$ should be $q$. If a weather model says "70% chance of rain" on 100 different days, approximately 70 of those days should actually see rain.

Calibration is about the **statistical reliability of the probabilities**, not the accuracy of the predictions. A model can be accurate but poorly calibrated (e.g., a model that correctly classifies 95% of inputs but assigns 99.9% confidence to all of them), or well-calibrated but inaccurate (e.g., a model that predicts 50% for everything on a balanced binary task — calibrated but useless).

### 7.2 Reliability Diagrams

A **reliability diagram** is the standard visual tool for assessing calibration. Construction:

1. Bin all predictions by their predicted confidence (e.g., 10 equal-width bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]).
2. For each bin, compute the **average predicted confidence** (x-axis) and the **actual accuracy** (y-axis).
3. Plot the result. A perfectly calibrated model falls on the diagonal $y = x$.

[FIG:ORIGINAL — Reliability diagram showing three calibration curves: well-calibrated (near the y=x diagonal), overconfident (below diagonal, typical of modern deep networks), and underconfident (above diagonal). Alongside, a histogram of sample counts per confidence bin showing that the overconfident model concentrates predictions at high confidence values.]

The following snippet constructs a reliability diagram and computes ECE from raw predictions:

```python
import numpy as np
import matplotlib.pyplot as plt

def reliability_diagram(confidences, accuracies, n_bins=10):
    """
    confidences: (N,) array of predicted probabilities for the chosen class
    accuracies:  (N,) boolean array — 1 if prediction was correct, 0 otherwise
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0); bin_confs.append((lo + hi) / 2); bin_counts.append(0)
            continue
        bin_accs.append(accuracies[mask].mean())
        bin_confs.append(confidences[mask].mean())
        bin_counts.append(mask.sum())

    bin_accs, bin_confs, bin_counts = map(np.array, [bin_accs, bin_confs, bin_counts])
    ece = (bin_counts / bin_counts.sum() * np.abs(bin_accs - bin_confs)).sum()

    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.bar(bin_confs, bin_counts / bin_counts.sum(), width=0.08, alpha=0.3, label='% of samples')
    ax1.set_ylabel('Fraction of samples')
    ax2 = ax1.twinx()
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    ax2.plot(bin_confs, bin_accs, 'o-', color='steelblue', lw=2, label=f'Model (ECE={ece:.3f})')
    ax2.set_ylabel('Accuracy'); ax2.set_xlabel('Confidence'); ax2.legend()
    plt.title('Reliability Diagram'); plt.tight_layout()
```

**Reading a reliability diagram:** Points below the diagonal mean overconfidence — the model claims higher probability than reality warrants. Points above mean underconfidence. Modern deep networks are typically overconfident: they produce sharp, near-one-hot softmax outputs even when the true posterior is more diffuse (Guo et al. 2017).

### 7.3 Expected Calibration Error (ECE)

ECE summarizes the reliability diagram as a single number:

$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} \left|\text{acc}(B_b) - \text{conf}(B_b)\right|$$

where $B_b$ is the set of predictions in bin $b$, $\text{acc}(B_b) = \frac{1}{|B_b|}\sum_{i \in B_b} \mathbf{1}[\hat{y}_i = y_i]$ is the accuracy in that bin, and $\text{conf}(B_b) = \frac{1}{|B_b|}\sum_{i \in B_b} \hat{p}_i$ is the mean predicted confidence. ECE is the weighted average gap between accuracy and confidence across all bins. Lower is better; zero is perfect calibration.

**Limitations of ECE:** The binning scheme is somewhat arbitrary — different bin widths or adaptive binning can change the number substantially. Kernel density-based calibration error (KDE-CE) and classwise calibration error are more principled alternatives, but ECE remains the most widely reported metric.

### 7.4 Why Modern Networks Are Miscalibrated

Guo et al. (2017) demonstrated that modern deep networks are significantly more miscalibrated than their predecessors. The primary culprits:

- **Increased model capacity.** Larger models can fit the training data more perfectly, pushing softmax outputs toward 0 and 1. The softmax function amplifies differences between logits exponentially — even a modest increase in the gap between the top logit and the second-highest produces a disproportionate increase in predicted confidence.
- **Batch normalization.** BN stabilizes training by normalizing hidden activations, enabling more training epochs and higher effective model capacity without divergence. The result is that models train longer and fit the training data more tightly, amplifying the overconfidence effect of increased capacity.
- **Overfitting to NLL.** Cross-entropy loss is a proper scoring rule (Section 8), but minimizing it on a finite training set can overfit the confidence — the model learns to be maximally confident on training examples, and this transfers to test examples even when it shouldn't.

### 7.5 Post-Hoc Calibration: Temperature Scaling

**Temperature scaling** (Guo et al. 2017) is the simplest and most effective post-hoc calibration method. It introduces a single scalar parameter $\tau > 0$ (the **temperature**) that rescales the logits before softmax:

$$\hat{\mathbf{p}}_{\text{calibrated}} = \text{softmax}(\mathbf{z} / \tau)$$

where $\mathbf{z}$ is the vector of logits (pre-softmax activations). The temperature is learned by minimizing NLL on a held-out **calibration set** (distinct from both training and test data):

$$\tau^* = \arg\min_\tau \; -\sum_{i=1}^{N_{\text{cal}}} \log \hat{p}_{\text{calibrated}}(y_i \mid \mathbf{x}_i; \tau)$$

**How temperature affects predictions:**

- **$\tau > 1$ (heating):** Divides logits by a number greater than 1, compressing them toward zero. The softmax output becomes softer (more uniform), reducing confidence. This corrects overconfidence. The name follows the statistical mechanics analogy: higher temperature means more disorder (a more uniform distribution over states).
- **$\tau < 1$ (cooling):** Amplifies logit differences, making the softmax output sharper. This corrects underconfidence. Lower temperature concentrates mass on the highest-energy (highest-logit) state.
- **$\tau = 1$:** No change — the original predictions.

**Critical property:** Temperature scaling does not change the argmax of the softmax — the predicted class label is unchanged. It only adjusts the confidence. This means accuracy is preserved exactly while calibration improves.

**Why temperature scaling works:** Neural networks are often overconfident because logit magnitudes grow larger than they need to be. A single scalar division corrects this global scale mismatch. The fact that a *single parameter* suffices suggests that the miscalibration is primarily a magnitude problem, not a shape problem — the relative ordering of logits is approximately correct, but their absolute scale is wrong.

**Other calibration methods:**

- **Platt scaling** (Platt 1999): Fit a logistic regression $\sigma(a \cdot z + b)$ on the top logit using the calibration set. More expressive than temperature scaling (two parameters: scale and shift) but limited to binary classification.
- **Isotonic regression:** Non-parametric monotone fit between predicted probabilities and observed frequencies. More flexible but prone to overfitting on small calibration sets.
- **Histogram binning:** Assign calibrated probabilities to bins directly. Simple but discontinuous.

In practice, temperature scaling dominates because it is simple, stable, and preserves accuracy.

### 7.6 Calibration for Regression

For regression, calibration means that predicted confidence intervals contain the true value at the stated frequency. A 90% prediction interval should contain the truth 90% of the time, across all predictions.

Checking this: for each confidence level $\alpha \in \{0.1, 0.2, \ldots, 0.9\}$, count the fraction of test points falling within the predicted $\alpha$-level interval. Plot the empirical coverage (y-axis) against the nominal coverage (x-axis). A calibrated model falls on the diagonal — just like the classification reliability diagram.

When the model predicts $\mathcal{N}(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))$ via a heteroscedastic output head (Section 6), the predicted variance $\sigma^2(\mathbf{x})$ may be well-ranked across inputs but systematically miscaled. The variance scaling method from Section 6.4 is the regression analogue of temperature scaling: it learns a single scalar $s^*$ on a calibration set that corrects the global scale of predicted intervals while preserving their relative ordering. Just as temperature scaling is cheap and never hurts for classification, variance scaling should be applied by default for regression.

Calibration asks whether predicted probabilities are trustworthy. Modern deep networks are systematically overconfident — their softmax outputs are sharper than they should be. ECE and reliability diagrams diagnose the problem; temperature scaling fixes it with a single learned parameter that rescales logits without changing predicted labels. For practical UQ, calibration is not optional — even well-designed uncertainty methods (ensembles, MC Dropout) benefit from post-hoc calibration.

---

## 8. Proper Scoring Rules: Evaluating Probabilistic Predictions

The previous sections produced probabilistic predictions; this section asks how to evaluate them. Accuracy alone is insufficient — it ignores confidence entirely. Proper scoring rules fill this gap by rewarding honest probability assessments. We define the concept, introduce the three most important rules (log loss, Brier score, and CRPS), and connect them to the losses already used in training.

### 8.1 The Problem: Accuracy Ignores Confidence

Consider two classifiers on the same test set:

| Input | True label | Model A | Model B |
|---|---|---|---|
| $x_1$ | cat | cat (51%) | cat (95%) |
| $x_2$ | dog | dog (51%) | dog (95%) |
| $x_3$ | cat | dog (51%) | cat (60%) |

Both Model A and Model B have 67% accuracy (2/3 correct). But Model B's correct predictions are confident and its incorrect prediction is less so — it "knows what it knows." Model A is barely distinguishable from random guessing. Accuracy cannot tell them apart; a proper scoring rule can.

### 8.2 Definition

A **scoring rule** $S(q, y)$ assigns a numerical score to a probabilistic prediction $q$ when the true outcome is $y$. A scoring rule is **proper** if the expected score is maximized (or minimized, depending on convention) when the predicted distribution $q$ equals the true distribution $p$:

$$\mathbb{E}_{Y \sim p}[S(p, Y)] \geq \mathbb{E}_{Y \sim p}[S(q, Y)] \qquad \text{for all } q$$

It is **strictly proper** if equality holds only when $q = p$.

**Intuition:** A proper scoring rule cannot be "gamed" — the best strategy is to report your true beliefs. If you think there's a 70% chance of rain, reporting 70% maximizes your expected score. Reporting 90% to seem more decisive, or 50% to hedge, both lower it.

### 8.3 The Big Three

**Log loss (negative log-likelihood):**

$$S_{\text{log}}(\mathbf{q}, y) = -\log q_y$$

where $q_y$ is the predicted probability of the true class $y$. This is the cross-entropy loss used to train most classifiers. It is strictly proper and heavily penalizes confident wrong predictions: predicting $q_y = 0.01$ for the true class costs $-\log(0.01) = 4.6$ nats, versus $-\log(0.5) = 0.69$ for an honest "I don't know."

**Brier score:**

$$S_{\text{Brier}}(\mathbf{q}, y) = \sum_{c=1}^C (q_c - \mathbf{1}[y = c])^2$$

For binary classification ($C = 2$), the sum has two terms: $(q_0 - \mathbf{1}[y=0])^2 + (q_1 - \mathbf{1}[y=1])^2$. Since $q_0 = 1 - q_1$ and $\mathbf{1}[y=0] = 1 - \mathbf{1}[y=1]$, both terms equal $(q_1 - \mathbf{1}[y=1])^2$, so the multi-class formula gives $2(q_1 - \mathbf{1}[y=1])^2$. Many references (including Brier's original 1950 paper) define the binary Brier score as the single-term form $(q_1 - \mathbf{1}[y=1])^2$ — half the multi-class formula. Both conventions are strictly proper; the factor of 2 does not affect ranking. In these notes, we use the single-term form $(q_1 - \mathbf{1}[y=1])^2$ for binary classification and the summation form $\sum_c (q_c - \mathbf{1}[y=c])^2$ for multi-class. Less sensitive to extreme predictions than log loss (bounded between 0 and 2 in the multi-class form, while log loss can go to $+\infty$).

The Brier score has a useful decomposition (Murphy 1973) into three terms. Partition the $N$ predictions into $B$ bins by predicted probability, with $n_b$ predictions in bin $b$, mean predicted probability $\bar{q}_b$, observed frequency $\bar{o}_b$, and overall base rate $\bar{o}$:

$$\text{Brier} = \underbrace{\frac{1}{N}\sum_{b=1}^B n_b\,(\bar{q}_b - \bar{o}_b)^2}_{\text{Reliability}} \;-\; \underbrace{\frac{1}{N}\sum_{b=1}^B n_b\,(\bar{o}_b - \bar{o})^2}_{\text{Resolution}} \;+\; \underbrace{\bar{o}\,(1 - \bar{o})}_{\text{Uncertainty}}$$

**Reliability** measures miscalibration — the weighted squared gap between predicted probabilities and observed frequencies in each bin. This is closely related to ECE (Section 7.3) but squared rather than absolute. Lower is better.

**Resolution** measures how much the model's predictions *differentiate* outcomes — how much the observed frequency in each bin deviates from the overall base rate. A model that predicts $\bar{o}$ for every input has zero resolution. Higher is better: the minus sign means higher resolution *decreases* the Brier score.

**Uncertainty** $= \bar{o}(1 - \bar{o})$ is a property of the dataset alone — it is maximized when the base rate is 50% and zero when one class dominates entirely. The model cannot influence it.

The diagnostic value: two models with the same Brier score may have very different profiles. A well-calibrated model with poor discrimination (low reliability, low resolution) and a sharply discriminating but miscalibrated model (high reliability, high resolution) can score identically — the decomposition reveals which problem to fix.

**Continuous Ranked Probability Score (CRPS):** For regression, where the prediction is a full distribution $F$ (CDF) and the observation is a scalar $y$:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left(F(z) - \mathbf{1}[z \geq y]\right)^2 dz$$

This is the Brier score generalized to continuous distributions. It measures the integrated squared difference between the predicted CDF and the step function at the observed value. A sharp, well-calibrated predictive distribution scores low; a diffuse or poorly-centred one scores high.

**Gaussian closed form.** For $F = \mathcal{N}(\mu, \sigma^2)$, the CRPS has a closed form (Gneiting & Raftery 2007):

$$\text{CRPS}(\mathcal{N}(\mu, \sigma^2),\, y) = \sigma\left[\frac{y - \mu}{\sigma}\left(2\Phi\!\left(\frac{y - \mu}{\sigma}\right) - 1\right) + 2\,\varphi\!\left(\frac{y - \mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}}\right]$$

where $\varphi$ and $\Phi$ are the standard normal PDF and CDF. A concrete comparison makes this tangible. Suppose we observe $y = 2.0$ and compare three predictive distributions:

| Prediction | $\mu$ | $\sigma$ | Standardized error $(y-\mu)/\sigma$ | CRPS |
|---|---|---|---|---|
| Sharp and correct | $2.0$ | $0.5$ | $0.0$ | $0.5\,(2 \cdot 0.399 - 1/\sqrt{\pi}) \approx 0.117$ |
| Diffuse and correct | $2.0$ | $2.0$ | $0.0$ | $2.0\,(2 \cdot 0.399 - 1/\sqrt{\pi}) \approx 0.469$ |
| Sharp but wrong | $0.0$ | $0.5$ | $4.0$ | $0.5\,(4.0 \cdot 0.999 + 2 \cdot 0.0001 - 1/\sqrt{\pi}) \approx 1.718$ |

The sharp, well-centred prediction scores best. Widening $\sigma$ while keeping $\mu$ correct (row 2) incurs a penalty for being diffuse — the model is honest but uninformative. Keeping $\sigma$ small but centring at the wrong value (row 3) is the worst outcome — the model is confidently wrong, which CRPS penalizes heavily.

### 8.4 Connection to Training Losses

The losses we routinely minimize during training are proper scoring rules:

| Training loss | Equivalent scoring rule | Proper? |
|---|---|---|
| Cross-entropy | Log loss (NLL) | Strictly proper |
| MSE (regression) | Brier score (binary case) | Strictly proper |
| Gaussian NLL (Section 6) | Log score for continuous distributions | Strictly proper |

This is not a coincidence. Training with a proper scoring rule ensures that the model's optimal strategy is to output the true conditional distribution $p(y \mid \mathbf{x})$, not some distorted version. If we trained with an improper scoring rule, the model's incentive would be to output something other than its true beliefs, making uncertainty estimates unreliable from the start.

### 8.5 Which Score to Report?

- **Log loss** is the default for classification — it is the training objective, it penalizes overconfidence sharply, and it decomposes into a calibration component and a refinement (sharpness) component.
- **Brier score** is preferred when you want a bounded metric that is less sensitive to rare, extreme miscalibrations.
- **CRPS** is the natural choice for regression with full predictive distributions — it evaluates the entire distribution, not just the mean or a single interval.
- **ECE** (Section 7.3) is *not* a proper scoring rule — a model can minimize ECE by strategically placing all predictions in a single bin. It is useful as a diagnostic but should not be used as the sole evaluation metric.

Proper scoring rules are the correct tool for evaluating probabilistic predictions. They reward honest probability assessments and cannot be gamed. Log loss and Brier score handle classification; CRPS handles regression. Conveniently, the losses we already use for training (cross-entropy, Gaussian NLL) are proper scoring rules, ensuring that optimal training produces calibrated predictions — at least in the infinite-data, well-specified limit. Finite data and model misspecification break this guarantee, which is why post-hoc calibration (Section 7) remains necessary in practice.

---

## 9. Practical Recommendations: Choosing a UQ Method

With the full toolkit in hand — epistemic methods (Sections 3–5), aleatoric modeling (Section 6), calibration (Section 7), and evaluation (Section 8) — we can now assemble a practical end-to-end pipeline. This section summarizes the method trade-offs, describes the recommended pipeline, and discusses how to choose when the full pipeline is too expensive.

### 9.1 Method Comparison

| Criterion | MC Dropout | Deep Ensembles | Heteroscedastic NLL | Laplace (last-layer) |
|---|---|---|---|---|
| **What it captures** | Epistemic | Epistemic + aleatoric (with NLL head) | Aleatoric only | Epistemic |
| **Training cost** | $1\times$ | $M\times$ | $1\times$ | $1\times$ (+ Hessian) |
| **Inference cost** | $T$ passes | $M$ passes | $1$ pass | $1$ pass (+ sampling) |
| **Code complexity** | Trivial | Low | Low | Moderate |
| **Calibration quality** | Moderate | Best | Good (aleatoric only) | Good |
| **Prerequisite** | Dropout in architecture | None | NLL loss | Trained model |

### 9.2 The Recommended Pipeline

**The recommended starting point for most applications** is a Deep Ensemble of $M = 5$ networks, each with a heteroscedastic (mean + variance) output head, followed by post-hoc calibration. The full pipeline:

1. **Architecture.** Use a shared-trunk architecture with two output heads: one for the predictive mean $\mu(\mathbf{x})$, one for the log-variance $\log \sigma^2(\mathbf{x})$ (Section 6.2).
2. **Training.** Train $M = 5$ copies of this architecture from different random initializations, each minimizing the Gaussian NLL loss (for regression) or cross-entropy (for classification). Same hyperparameters, same data, different seeds.
3. **Inference.** For a new input $\mathbf{x}_*$, collect $(\mu_m(\mathbf{x}_*), \sigma^2_m(\mathbf{x}_*))$ from each member. Compute the ensemble predictive mean and the total variance via the law of total variance (Section 5.1):

$$\bar{\mu} = \frac{1}{M}\sum_m \mu_m, \qquad \sigma^2_{\text{total}} = \underbrace{\frac{1}{M}\sum_m \sigma^2_m}_{\text{aleatoric}} + \underbrace{\frac{1}{M}\sum_m (\mu_m - \bar{\mu})^2}_{\text{epistemic}}$$

4. **Calibration.** Hold out a calibration set (distinct from training and test). For classification, learn a temperature $\tau^*$ by minimizing NLL on the calibration set (Section 7.5). For regression, learn a variance scale $s^*$ (Section 6.4). Apply at test time.
5. **Reporting.** Report the predictive mean, the total uncertainty (e.g., $\pm 2\sigma_{\text{total}}$ intervals for regression, predictive entropy for classification), and — where actionable — the aleatoric/epistemic decomposition separately.

This pipeline gives both types of uncertainty, is simple to implement, and consistently produces the best empirical calibration across benchmarks.

### 9.3 The Pipeline in Action: Ensemble vs. MC Dropout on 1D Heteroscedastic Data

The following self-contained example puts the full pipeline side-by-side with MC Dropout on a problem designed to stress-test UQ: a 1D regression task with **heteroscedastic noise** (noise grows with $x$) and an **extrapolation region** (no training data for $x > 3$). The comparison makes the "ensembles capture more epistemic uncertainty than dropout" claim from Section 5.3 visually concrete.

**Data.** The true function is $\sin(x)$ with noise standard deviation $0.05 + 0.15|x|$ — small near the origin, substantial at the edges. Training data covers $x \in [-1, 3]$; the test grid extends to $x = 6$ to probe extrapolation.

```python
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
N = 200
x_train = np.random.uniform(-1, 3, N).astype(np.float32)
noise_std = 0.05 + 0.15 * np.abs(x_train)
y_train = np.sin(x_train) + noise_std * np.random.randn(N).astype(np.float32)
X, Y = torch.tensor(x_train).unsqueeze(1), torch.tensor(y_train).unsqueeze(1)
x_grid = torch.linspace(-2, 6, 400).unsqueeze(1)
```

**Model.** A two-hidden-layer network with a heteroscedastic output head (Section 6). A `use_dropout` flag toggles between the ensemble variant (no dropout) and the MC Dropout variant. Both are trained with the Gaussian NLL loss from Section 6.2.

```python
class HetNet(nn.Module):
    def __init__(self, use_dropout=False, p=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.drop = nn.Dropout(p) if use_dropout else nn.Identity()
        self.mu_head, self.logvar_head = nn.Linear(64, 1), nn.Linear(64, 1)

    def forward(self, x):
        h = self.drop(self.trunk(x))
        return self.mu_head(h), self.logvar_head(h)

def nll_loss(y, mu, logvar):
    return (0.5 * ((y - mu)**2 / torch.exp(logvar) + logvar)).mean()

def train_model(model, X, Y, epochs=1500, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        mu, logvar = model(X)
        opt.zero_grad(); nll_loss(Y, mu, logvar).backward(); opt.step()
```

**Training.** Five ensemble members from different random seeds, plus one MC Dropout model.

```python
M = 5
ensemble = []
for m in range(M):
    torch.manual_seed(m * 17 + 42)
    net = HetNet(use_dropout=False); train_model(net, X, Y); ensemble.append(net)

torch.manual_seed(999)
mc_net = HetNet(use_dropout=True, p=0.1)
train_model(mc_net, X, Y)
```

**Inference.** For the ensemble, the law of total variance (Section 5.1) gives aleatoric and epistemic components directly. For MC Dropout, $T = 50$ stochastic forward passes play the same role.

```python
with torch.no_grad():
    mus = torch.stack([net(x_grid)[0] for net in ensemble])
    logvars = torch.stack([net(x_grid)[1] for net in ensemble])
    vars_alea = torch.exp(logvars).mean(dim=0).squeeze()
    mu_bar = mus.mean(dim=0).squeeze()
    vars_epist = ((mus.squeeze() - mu_bar)**2).mean(dim=0)
    ens_total = (vars_alea + vars_epist).sqrt().numpy()
    ens_alea = vars_alea.sqrt().numpy()

T = 50; mc_net.train()
with torch.no_grad():
    mc_mus = torch.stack([mc_net(x_grid)[0] for _ in range(T)])
    mc_logvars = torch.stack([mc_net(x_grid)[1] for _ in range(T)])
mc_mu_bar = mc_mus.mean(dim=0).squeeze().numpy()
mc_alea = torch.exp(mc_logvars).mean(dim=0).squeeze().sqrt().numpy()
mc_epist = mc_mus.squeeze().var(dim=0).sqrt().numpy()
mc_total = np.sqrt(mc_alea**2 + mc_epist**2)
```

**Plotting.** Side-by-side comparison with orange aleatoric bands and blue total-uncertainty bands.

```python
xg = x_grid.squeeze().numpy()
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, mu_np, total, alea, title in [
    (axes[0], mu_bar.numpy(), ens_total, ens_alea, f'Deep Ensemble (M={M})'),
    (axes[1], mc_mu_bar,      mc_total,  mc_alea,  f'MC Dropout (T={T})'),
]:
    ax.scatter(x_train, y_train, s=6, alpha=0.3, color='gray', label='Training data')
    ax.plot(xg, mu_np, 'steelblue', lw=2, label='Predictive mean')
    ax.fill_between(xg, mu_np-2*total, mu_np+2*total, alpha=0.15, color='steelblue', label='±2σ total')
    ax.fill_between(xg, mu_np-2*alea, mu_np+2*alea, alpha=0.3, color='orange', label='±2σ aleatoric')
    ax.axvspan(3, 6, alpha=0.07, color='red', label='Extrapolation region')
    ax.set_xlabel('x'); ax.set_title(title); ax.legend(fontsize=8)
axes[0].set_ylabel('y')
plt.suptitle('Ensemble vs. MC Dropout: uncertainty on heteroscedastic 1D data', y=1.02)
plt.tight_layout(); plt.savefig('images/ensemble_vs_dropout_1d.png', dpi=150, bbox_inches='tight')
```

[FIG:ORIGINAL — Side-by-side comparison of Deep Ensemble (M=5, left) and MC Dropout (T=50, right) on 1D heteroscedastic regression data with training range x∈[-1,3] and extrapolation region x>3. Shows orange aleatoric bands and blue total-uncertainty bands; the ensemble's blue band fans out dramatically in the extrapolation region while MC Dropout's band widens much less.]

**What to look for in the plot:**

- **Within the training range ($x \in [-1, 3]$):** Both methods produce similar aleatoric bands that widen with $x$, correctly tracking the heteroscedastic noise. The epistemic component (gap between the orange and blue bands) is small for both — the model has seen enough data here.
- **In the extrapolation region ($x > 3$):** The ensemble's blue band fans out substantially — different ensemble members extrapolate differently, producing large epistemic variance. MC Dropout's blue band widens much less, because all $T$ forward passes share the same converged weights and only vary by dropout mask — they explore the *neighbourhood* of a single solution, not different modes.
- **The aleatoric band continues to grow** in the extrapolation region for both methods. This is the heteroscedastic head doing its job — it learned that noise scales with $|x|$ and continues to predict that pattern. But this is pure extrapolation of the learned noise model; the aleatoric estimate in the extrapolation region is less trustworthy than within the training range.

This is the key empirical argument from Section 5.3 made visual: inter-mode diversity (ensembles) produces wider, more honest uncertainty bands in extrapolation than intra-mode perturbation (dropout).

### 9.4 When the Full Pipeline Is Too Expensive

If training $M$ models is infeasible:

- **MC Dropout** is the cheapest epistemic uncertainty method — assuming the architecture already uses dropout. Combine with a heteroscedastic output head for aleatoric uncertainty. Be aware that the dropout rate should ideally be tuned for uncertainty quality (validation log-likelihood), not just regularization performance.
- **Last-layer Laplace** requires only a single trained model and a one-time Hessian computation over the final layer. It is particularly attractive for pre-trained models where retraining is not an option: freeze the feature extractor, compute the Hessian of the last linear layer (which is just BLR), and sample at test time. Libraries like `laplace-torch` (Daxberger et al. 2021) make this a few lines of code.
- **Snapshot ensembles** (Huang et al. 2017) use cyclical learning rate schedules to visit multiple modes during a single training run, saving checkpoints at each cycle's minimum. This gives ensemble-like diversity at $\sim 1\times$ training cost, though with less diversity than independent initializations.

### 9.5 A Decision Checklist

1. **Do you need to distinguish aleatoric from epistemic?** If yes, you need both a heteroscedastic output head (aleatoric) and an ensemble or MC Dropout (epistemic). If you only need total uncertainty, a simpler setup may suffice.
2. **Is training budget the bottleneck?** If yes, use MC Dropout or last-layer Laplace. If no, use ensembles.
3. **Is inference latency the bottleneck?** If yes, last-layer Laplace or a single heteroscedastic model is cheapest ($1$ forward pass). If latency is flexible, ensembles ($M$ passes) or MC Dropout ($T$ passes) are fine.
4. **Will the model encounter distribution shift at deployment?** Ensembles degrade most gracefully under shift (Ovadia et al. 2019). MC Dropout can underestimate uncertainty on shifted data.
5. **Always calibrate post-hoc.** Regardless of which method you choose, apply temperature scaling (classification) or variance scaling (regression) on a held-out calibration set. This is cheap, never hurts, and often produces the single largest improvement in calibration quality.

The gold-standard UQ pipeline is a heteroscedastic deep ensemble with post-hoc calibration — it captures both uncertainty types, requires no exotic infrastructure, and wins empirical benchmarks. When that is too expensive, MC Dropout and last-layer Laplace offer principled alternatives at lower cost. The key meta-lesson: no single method is universally best, but *any* principled uncertainty method, properly calibrated, is vastly better than ignoring uncertainty entirely.

**Beyond the reading: Conformal prediction — distribution-free UQ.** All methods in Sections 3–7 rely on distributional assumptions (Gaussian likelihoods, specific variational families, softmax calibration). **Conformal prediction** (Vovk et al. 2005) takes a fundamentally different approach: it provides *finite-sample, distribution-free* coverage guarantees. The core idea: given a calibration set of $n$ examples and a new test input, conformal prediction constructs a prediction set $\mathcal{C}(\mathbf{x}_*)$ such that $P(y_* \in \mathcal{C}(\mathbf{x}_*)) \geq 1 - \alpha$, where $\alpha$ is a user-chosen error rate (e.g., 0.1 for 90% coverage). This guarantee holds for *any* model and *any* data distribution, assuming only that calibration and test data are exchangeable (a weaker assumption than i.i.d.). For regression, the prediction set is an interval; for classification, it is a subset of classes.

The practical algorithm is simple. For regression with a pre-trained model $\hat{f}$:

1. Compute the nonconformity score $s_i = |y_i - \hat{f}(\mathbf{x}_i)|$ (the absolute residual) for each of the $n$ calibration examples.
2. Sort the scores and take $\hat{q}$ as the $\lceil(1-\alpha)(n+1)\rceil / n$ quantile — this is the smallest threshold that covers at least a $(1-\alpha)$ fraction of calibration points (with a finite-sample correction).
3. For a new test input, form $\mathcal{C}(\mathbf{x}_*) = [\hat{f}(\mathbf{x}_*) - \hat{q},\; \hat{f}(\mathbf{x}_*) + \hat{q}]$.

**Concrete example:** Suppose $n = 9$ calibration points with sorted absolute residuals $[0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5]$ and $\alpha = 0.1$ (90% coverage). Then $\lceil 0.9 \times 10 \rceil = 9$, and the $9/9 = 100$th percentile of the scores is $\hat{q} = 1.5$. For a new input with $\hat{f}(\mathbf{x}_*) = 3.0$, the prediction interval is $[1.5,\, 4.5]$. The interval is wide here because $n$ is small — with more calibration data, $\hat{q}$ would tighten toward the true 90th-percentile residual.

Conformal prediction does not replace the Bayesian methods above — it says nothing about *why* the model is uncertain (aleatoric vs. epistemic) — but it provides a rigorous coverage guarantee that model-based methods cannot match without correct distributional assumptions.

**Beyond the reading: Out-of-distribution (OOD) detection.** Epistemic uncertainty methods have a direct application beyond error bars: detecting inputs that are *out of distribution* — inputs the model should not be trusted on at all. The idea is natural: if a test input $\mathbf{x}_*$ is far from the training distribution, ensemble members will disagree strongly (high epistemic variance) and MC Dropout forward passes will produce high-entropy predictions.

A simple OOD detector thresholds an epistemic uncertainty metric (e.g., mutual information for classification, ensemble variance for regression) and flags inputs above the threshold for human review or rejection. This connects UQ directly to deployment safety: rather than producing a (possibly meaningless) prediction on an alien input, the system says "I don't know enough to answer."

Hendrycks & Gimpel (2017) showed that even simple baselines (maximum softmax probability) have some OOD detection power; epistemic uncertainty from ensembles or MC Dropout substantially improves it by separating "confused because ambiguous" (aleatoric — softmax entropy is high but stable across members) from "confused because unfamiliar" (epistemic — softmax entropy varies across members).

---

## Sources and Further Reading

- Gal, Y. and Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." *ICML 2016*. The theoretical foundation for MC Dropout.
- Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS 2017*. The Deep Ensembles paper — establishes ensembles as the empirical gold standard.
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). "On Calibration of Modern Neural Networks." *ICML 2017*. Demonstrates systematic overconfidence in deep networks and introduces temperature scaling.
- Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra, D. (2015). "Weight Uncertainty in Neural Networks." *ICML 2015*. Bayes by Backprop — variational inference for neural network weights.
- Kendall, A. and Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" *NeurIPS 2017*. Combines aleatoric and epistemic uncertainty; introduces the heteroscedastic + MC Dropout framework.
- Ovadia, Y. et al. (2019). "Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift." *NeurIPS 2019*. Large-scale comparison of UQ methods under distribution shift — ensembles win.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." *JMLR 15*. The original dropout paper.
- Daxberger, E. et al. (2021). "Laplace Redux — Effortless Bayesian Deep Learning." *NeurIPS 2021*. Last-layer and subnetwork Laplace approximations for practical BDL.
- Martens, J. and Grosse, R. (2015). "Optimizing Neural Networks with Kronecker-Factored Approximate Curvature." *ICML 2015*. KFAC — the Kronecker-factored Hessian approximation used in scalable Laplace methods.
- Welling, M. and Teh, Y. W. (2011). "Bayesian Learning via Stochastic Gradient Langevin Dynamics." *ICML 2011*. Stochastic gradient MCMC — combines SGD with Langevin noise for approximate posterior sampling.
- Gneiting, T. and Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association 102(477)*. The definitive reference on proper scoring rules.
- Bishop, C. M. *Pattern Recognition and Machine Learning* (2006). Chapter 3 (Bayesian linear regression), Chapter 5 (neural networks). The Bayesian linear regression predictive variance (equation 3.59) is the closed-form special case that this document generalizes.
- Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. The foundational text on conformal prediction — distribution-free coverage guarantees for any predictive model.
- Hendrycks, D. and Gimpel, K. (2017). "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks." *ICLR 2017*. Establishes maximum softmax probability as a simple OOD detection baseline.
- Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., and Weinberger, K. Q. (2017). "Snapshot Ensembles: Train 1, Get M for Free." *ICLR 2017*. Cyclical learning rates to collect multiple models from a single training run.
