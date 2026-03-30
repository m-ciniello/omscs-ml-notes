# Quadratic Forms: A Reference for ML

*OMSCS ML | Linear Algebra Foundations*

---

## Table of Contents

1. [What Is a Quadratic Form?](#1-what-is-a-quadratic-form)
2. [The Symmetry Convention](#2-the-symmetry-convention)
3. [Geometric Interpretation: Level Sets and Ellipsoids](#3-geometric-interpretation)
4. [Definiteness](#4-definiteness)
5. [The Eigendecomposition and the Principal Axes Form](#5-eigendecomposition)
6. [The Eigenvalue Sandwich Inequality](#6-eigenvalue-sandwich)
7. [The Rayleigh Quotient](#7-rayleigh-quotient)
8. [The Condition Number](#8-condition-number)
9. [Completing the Square for Quadratics with Linear Terms](#9-completing-the-square)
10. [Applications in ML](#10-applications)
   - 10.1 The Sample Covariance Matrix as a Quadratic Form
   - 10.2 Local Quadratic Approximation of a Loss
   - 10.3 PCA: Variance as a Rayleigh Quotient
   - 10.4 Ridge Regression
   - 10.5 Mahalanobis Distance
   - 10.6 Gradient Descent Analysis: The L-Smoothness Descent Lemma
11. [Sources and Further Reading](#11-sources)

---

## 1. What Is a Quadratic Form?

### Definition

A **quadratic form** is a scalar-valued function of a vector $\mathbf{u} \in \mathbb{R}^n$ that is quadratic in each component — meaning it contains only degree-2 terms (no linear terms, no constants). Given a matrix $S \in \mathbb{R}^{n \times n}$, the quadratic form associated with $S$ is:

$$Q(\mathbf{u}) = \mathbf{u}^T S \mathbf{u}$$

To see what this expands to, write it out explicitly:

$$\mathbf{u}^T S \mathbf{u} = \sum_{i=1}^{n} \sum_{j=1}^{n} S_{ij} \, u_i \, u_j$$

Every term is a product of exactly two components of $\mathbf{u}$ (possibly the same component twice), scaled by $S_{ij}$. No term is just $u_i$ alone (that would be degree 1), and no constant term appears. This is why it is called *quadratic*.

### A Concrete 2D Example

Take $\mathbf{u} = (u_1, u_2)^T$ and $S = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$. Then:

$$\mathbf{u}^T S \mathbf{u} = \begin{pmatrix} u_1 & u_2 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix} \begin{pmatrix} u_1 \\ u_2 \end{pmatrix}$$

First multiply $\mathbf{u}^T S$:

$$\mathbf{u}^T S = \begin{pmatrix} 3u_1 + u_2, \; u_1 + 2u_2 \end{pmatrix}$$

Then take the dot product with $\mathbf{u}$:

$$\mathbf{u}^T S \mathbf{u} = (3u_1 + u_2)u_1 + (u_1 + 2u_2)u_2 = 3u_1^2 + u_1 u_2 + u_1 u_2 + 2u_2^2 = 3u_1^2 + 2u_1 u_2 + 2u_2^2$$

Notice the cross-term $2 u_1 u_2$. It came from two off-diagonal contributions: $S_{12} u_1 u_2 + S_{21} u_2 u_1 = 1 \cdot u_1 u_2 + 1 \cdot u_1 u_2$.

---

## 2. The Symmetry Convention

### Why We Always Work With Symmetric Matrices

You might wonder: does the matrix $S$ in a quadratic form need to be symmetric? It turns out it doesn't need to be, but we can *always assume* it is without changing the value of the form. Here's why.

**Claim:** For any matrix $S$, the quadratic form $\mathbf{u}^T S \mathbf{u}$ depends only on the symmetric part $\frac{1}{2}(S + S^T)$.

**Proof:** Since $\mathbf{u}^T S \mathbf{u}$ is a scalar, it equals its own transpose:

$$\mathbf{u}^T S \mathbf{u} = (\mathbf{u}^T S \mathbf{u})^T = \mathbf{u}^T S^T \mathbf{u}$$

Adding these two equal quantities and dividing by 2:

$$\mathbf{u}^T S \mathbf{u} = \frac{1}{2}\mathbf{u}^T S \mathbf{u} + \frac{1}{2}\mathbf{u}^T S^T \mathbf{u} = \mathbf{u}^T \underbrace{\frac{S + S^T}{2}}_{\text{symmetric}} \mathbf{u}$$

The antisymmetric part $\frac{S - S^T}{2}$ drops out completely — it contributes zero to every quadratic form.

**Practical implication:** In ML, whenever you see $\mathbf{u}^T S \mathbf{u}$, the matrix $S$ is almost always already symmetric. Covariance matrices, Hessians of smooth losses, kernel matrices, and the matrix $X^T X$ in regression are all symmetric by construction. We will assume $S = S^T$ throughout.

---

## 3. Geometric Interpretation: Level Sets and Ellipsoids

### What Does $\mathbf{u}^T S \mathbf{u} = c$ Look Like?

The level set of a quadratic form is the set of all $\mathbf{u}$ that produce the same value $c$:

$$\{ \mathbf{u} \in \mathbb{R}^n : \mathbf{u}^T S \mathbf{u} = c \}$$

For a positive definite matrix $S$ (defined precisely in Section 4), this is an **ellipsoid** centered at the origin. In 2D it is an ellipse; in 3D an ellipsoid; in general $n$ dimensions an $(n-1)$-dimensional ellipsoidal surface.

### Why an Ellipsoid?

The simplest case is $S = I$ (the identity matrix). Then $\mathbf{u}^T I \mathbf{u} = \|{\mathbf{u}}\|^2 = c$, which is a sphere of radius $\sqrt{c}$. The quadratic form just measures squared Euclidean distance from the origin.

Now scale each axis differently: $S = \text{diag}(s_1, s_2)$ with $s_1 \neq s_2$. Then:

$$s_1 u_1^2 + s_2 u_2^2 = c \quad \Longleftrightarrow \quad \frac{u_1^2}{c/s_1} + \frac{u_2^2}{c/s_2} = 1$$

This is an ellipse with semi-axes $\sqrt{c/s_1}$ and $\sqrt{c/s_2}$. Large eigenvalue → small semi-axis (tightly curved). Small eigenvalue → large semi-axis (gently curved). The ellipse is elongated in the direction where $S$ "pushes back" least.

For a general symmetric $S$ with off-diagonal terms, the ellipse is the same thing but *rotated* — the axes of the ellipse are the eigenvectors of $S$, and the lengths are determined by the eigenvalues. We'll make this precise in Section 5.

### Why This Matters for Gradient Descent

When you Taylor-expand a loss $L$ near a minimum $\boldsymbol{\theta}^*$, the local approximation is:

$$L(\boldsymbol{\theta}^* + \mathbf{u}) \approx L(\boldsymbol{\theta}^*) + \frac{1}{2} \mathbf{u}^T H \mathbf{u}$$

where $H$ is the Hessian. The **level sets of this approximation are ellipsoids** — the "bowl" you're descending is an ellipsoidal bowl. If the bowl is nearly spherical (eigenvalues of $H$ all similar), gradient descent works great. If it's highly elongated (some eigenvalues much larger than others), gradient descent struggles. The quadratic form is what gives the bowl its shape.

---

## 4. Definiteness

### Motivation

Given a symmetric matrix $S$, the sign behavior of $\mathbf{u}^T S \mathbf{u}$ across all possible $\mathbf{u} \neq \mathbf{0}$ tells us the shape of the associated quadratic surface. This is so important it gets its own taxonomy.

### The Five Cases

Let $S$ be a symmetric $n \times n$ matrix. We say:

| Name | Condition | Meaning |
|------|-----------|---------|
| **Positive definite (PD)** | $\mathbf{u}^T S \mathbf{u} > 0$ for all $\mathbf{u} \neq \mathbf{0}$ | Strictly bowl-shaped; unique minimum at origin |
| **Positive semidefinite (PSD)** | $\mathbf{u}^T S \mathbf{u} \geq 0$ for all $\mathbf{u}$ | Bowl-shaped but flat in some directions |
| **Negative definite (ND)** | $\mathbf{u}^T S \mathbf{u} < 0$ for all $\mathbf{u} \neq \mathbf{0}$ | Strictly inverted bowl; unique maximum at origin |
| **Negative semidefinite (NSD)** | $\mathbf{u}^T S \mathbf{u} \leq 0$ for all $\mathbf{u}$ | Inverted bowl, flat in some directions |
| **Indefinite** | Takes both positive and negative values | Saddle surface |

### The Eigenvalue Characterization

The definiteness of $S$ is completely determined by its eigenvalues. Since $S$ is symmetric, all eigenvalues are real.

- $S$ is **PD** $\Leftrightarrow$ all eigenvalues $> 0$
- $S$ is **PSD** $\Leftrightarrow$ all eigenvalues $\geq 0$
- $S$ is **ND** $\Leftrightarrow$ all eigenvalues $< 0$
- $S$ is **NSD** $\Leftrightarrow$ all eigenvalues $\leq 0$
- $S$ is **indefinite** $\Leftrightarrow$ some eigenvalues $> 0$ and some $< 0$

We'll prove the PD case in Section 5 once we have the eigendecomposition in hand. For now, take it as the key fact: **definiteness is the sign of the spectrum.**

### Geometric Intuition

Think of a 2D example. If both eigenvalues are positive, the quadratic surface $z = \mathbf{u}^T S \mathbf{u}$ is a bowl (paraboloid opening upward). If one eigenvalue is positive and one is negative, the surface is a saddle — goes up in one direction and down in another. If both eigenvalues are zero, the surface is flat (constant zero). The eigenvalues literally control the curvature in each principal direction.

### Why PD Matters in ML

The Hessian of a strictly convex loss is positive definite. This is what guarantees a unique global minimum. Ridge regression adds $\lambda \|\boldsymbol{\theta}\|^2$ to the loss precisely to ensure the modified Hessian $X^T X + \lambda I$ is strictly positive definite even when $X^T X$ is only PSD.

---

## 5. The Eigendecomposition and the Principal Axes Form

### Setup

Since $S$ is real and symmetric, the spectral theorem guarantees it can be diagonalized by an orthonormal basis of eigenvectors. Concretely:

$$S = Q \Lambda Q^T$$

where:
- $Q = [\mathbf{q}_1 \mid \mathbf{q}_2 \mid \cdots \mid \mathbf{q}_n]$ is orthogonal: columns are eigenvectors, $Q^T Q = Q Q^T = I$
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ contains the corresponding real eigenvalues

### Substituting Into the Quadratic Form

Substitute $S = Q \Lambda Q^T$:

$$\mathbf{u}^T S \mathbf{u} = \mathbf{u}^T Q \Lambda Q^T \mathbf{u}$$

Now define the **rotated coordinates** $\tilde{\mathbf{u}} = Q^T \mathbf{u}$. Since $Q$ is orthogonal, this is just a rotation (no stretching) — it expresses $\mathbf{u}$ in the coordinate system aligned with the eigenvectors of $S$.

To substitute, note that $Q^T\mathbf{u} = \tilde{\mathbf{u}}$, and taking the transpose of both sides, $(Q^T\mathbf{u})^T = \tilde{\mathbf{u}}^T$, which expands as $\mathbf{u}^T Q = \tilde{\mathbf{u}}^T$ (using $(AB)^T = B^T A^T$ and $(Q^T)^T = Q$). So $\mathbf{u}^T Q = \tilde{\mathbf{u}}^T$ and $Q^T\mathbf{u} = \tilde{\mathbf{u}}$, giving:

$$\mathbf{u}^T Q \Lambda Q^T \mathbf{u} = \tilde{\mathbf{u}}^T \Lambda \tilde{\mathbf{u}}$$

Since $\Lambda$ is diagonal, this telescopes beautifully:

$$\tilde{\mathbf{u}}^T \Lambda \tilde{\mathbf{u}} = \sum_{i=1}^{n} \lambda_i \tilde{u}_i^2$$

**So what?** In the eigenbasis, the quadratic form is just a *weighted sum of squares* with weights equal to the eigenvalues. The off-diagonal "cross terms" vanish entirely. This is the **principal axes form** of the quadratic.

### What This Proves About Definiteness

From the principal axes form, the sign of $\mathbf{u}^T S \mathbf{u}$ is determined entirely by the signs of the $\lambda_i$:

- If all $\lambda_i > 0$: every term $\lambda_i \tilde{u}_i^2 \geq 0$, and at least one is positive when $\tilde{\mathbf{u}} \neq \mathbf{0}$ (since $Q$ is invertible, $\mathbf{u} \neq \mathbf{0} \Rightarrow \tilde{\mathbf{u}} \neq \mathbf{0}$). So the sum is $> 0$. Hence $S$ is PD.
- If any $\lambda_k < 0$: set $\tilde{\mathbf{u}} = \mathbf{e}_k$ (the $k$-th standard basis vector), so $\tilde{u}_k = 1$ and all other $\tilde{u}_i = 0$. Then $\sum_i \lambda_i \tilde{u}_i^2 = \lambda_k < 0$, showing the form can be negative. Similarly you can find $\tilde{\mathbf{u}}$ making it positive. So $S$ is indefinite unless all eigenvalues share the same sign.

This completes the justification of the eigenvalue characterization stated in Section 4.

### The Ellipsoid Axis Lengths

The level set $\mathbf{u}^T S \mathbf{u} = 1$ (for PD $S$) satisfies in the rotated frame:

$$\sum_{i=1}^{n} \lambda_i \tilde{u}_i^2 = 1 \quad \Longleftrightarrow \quad \sum_{i=1}^{n} \frac{\tilde{u}_i^2}{1/\lambda_i} = 1$$

This is an axis-aligned ellipsoid in the $\tilde{\mathbf{u}}$ frame, with semi-axis length $1/\sqrt{\lambda_i}$ along the $i$-th eigenvector direction. In the original $\mathbf{u}$ frame, the ellipsoid is rotated by $Q$ but otherwise the same shape.

**Key takeaway:** The eigenvectors of $S$ point along the axes of the ellipsoid; the eigenvalues control the axis lengths (inversely). Large eigenvalue → short axis (tightly curved); small eigenvalue → long axis (gently curved).

---

## 6. The Eigenvalue Sandwich Inequality

### Statement

Let $S$ be a symmetric PSD matrix with eigenvalues $0 \leq \lambda_{\min} \leq \cdots \leq \lambda_{\max}$. Then for all $\mathbf{u} \in \mathbb{R}^n$:

$$\lambda_{\min} \|\mathbf{u}\|^2 \leq \mathbf{u}^T S \mathbf{u} \leq \lambda_{\max} \|\mathbf{u}\|^2$$

### Derivation

Using the principal axes form from Section 5 and the fact that $Q$ is orthogonal (so $\|\tilde{\mathbf{u}}\|^2 = \|\mathbf{u}\|^2$):

**Upper bound.** Since $\lambda_i \leq \lambda_{\max}$ for all $i$:

$$\mathbf{u}^T S \mathbf{u} = \sum_{i=1}^{n} \lambda_i \tilde{u}_i^2 \leq \lambda_{\max} \sum_{i=1}^{n} \tilde{u}_i^2 = \lambda_{\max} \|\tilde{\mathbf{u}}\|^2 = \lambda_{\max} \|\mathbf{u}\|^2$$

**Lower bound.** Since $\lambda_i \geq \lambda_{\min}$ for all $i$:

$$\mathbf{u}^T S \mathbf{u} = \sum_{i=1}^{n} \lambda_i \tilde{u}_i^2 \geq \lambda_{\min} \sum_{i=1}^{n} \tilde{u}_i^2 = \lambda_{\min} \|\mathbf{u}\|^2$$

**When are the bounds tight?** The upper bound holds with equality when all weight in $\sum_i \lambda_i \tilde{u}_i^2$ is concentrated on the index achieving $\lambda_{\max}$: set $\tilde{\mathbf{u}} = \mathbf{e}_k$ where $\lambda_k = \lambda_{\max}$. In the original frame this means $\mathbf{u} = Q\mathbf{e}_k = \mathbf{q}_{\max}$ — the eigenvector for $\lambda_{\max}$. Similarly the lower bound is tight at $\mathbf{u} = \mathbf{q}_{\min}$. So the eigenvalues are not just bounds; they are *achieved* values.

### So What?

This inequality is the workhorse behind convergence proofs in optimization. Its uses include:

**1. The L-smoothness condition.** A function is called $L$-smooth (or has $L$-Lipschitz gradient) if its Hessian is bounded above: $H \preceq L \cdot I$, meaning $\mathbf{u}^T H \mathbf{u} \leq L \|\mathbf{u}\|^2$ for all $\mathbf{u}$. The sandwich tells us this is equivalent to $\lambda_{\max}(H) \leq L$. This condition is what lets you prove that gradient descent with step size $\alpha \leq 1/L$ decreases the loss every step.

**2. The strong convexity condition.** A function is $m$-strongly convex if $H \succeq m \cdot I$, i.e., $\mathbf{u}^T H \mathbf{u} \geq m \|\mathbf{u}\|^2$. The sandwich says this is $\lambda_{\min}(H) \geq m > 0$. Strong convexity gives linear convergence rate guarantees.

**3. Bounding inner products.** Sometimes you need to bound $|\mathbf{u}^T S \mathbf{v}|$ for two different vectors. The sandwich, combined with Cauchy-Schwarz, lets you do this in terms of the spectrum of $S$.

---

## 7. The Rayleigh Quotient

### Motivation

The sandwich from Section 6 says $\lambda_{\min} \|\mathbf{u}\|^2 \leq \mathbf{u}^T S \mathbf{u} \leq \lambda_{\max} \|\mathbf{u}\|^2$. This naturally raises the question: what is the *maximum* of $\mathbf{u}^T S \mathbf{u}$ subject to $\|\mathbf{u}\| = 1$? And what direction achieves it?

This is the Rayleigh quotient problem, and its answer is central to PCA.

### Definition

The **Rayleigh quotient** of a symmetric matrix $S$ for a nonzero vector $\mathbf{u}$ is:

$$R(\mathbf{u}) = \frac{\mathbf{u}^T S \mathbf{u}}{\mathbf{u}^T \mathbf{u}} = \frac{\mathbf{u}^T S \mathbf{u}}{\|\mathbf{u}\|^2}$$

Note that $R(\mathbf{u})$ is **scale-invariant** — replacing $\mathbf{u}$ by $c\mathbf{u}$ for any scalar $c \neq 0$ gives the same value (numerator scales by $c^2$, denominator scales by $c^2$). So maximizing $R(\mathbf{u})$ over all nonzero $\mathbf{u}$ is equivalent to maximizing $\mathbf{u}^T S \mathbf{u}$ subject to $\|\mathbf{u}\| = 1$.

### The Rayleigh-Ritz Theorem

**Theorem.** For a symmetric matrix $S$ with eigenvalues $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$ and corresponding orthonormal eigenvectors $\mathbf{q}_1, \ldots, \mathbf{q}_n$:

$$\max_{\mathbf{u} \neq \mathbf{0}} R(\mathbf{u}) = \lambda_{\max}, \quad \text{achieved at } \mathbf{u} = \mathbf{q}_{\max}$$

$$\min_{\mathbf{u} \neq \mathbf{0}} R(\mathbf{u}) = \lambda_{\min}, \quad \text{achieved at } \mathbf{u} = \mathbf{q}_{\min}$$

More generally, the $k$-th eigenvalue is the maximum of $R(\mathbf{u})$ over all $\mathbf{u}$ orthogonal to the first $k-1$ eigenvectors (the Courant-Fischer minimax theorem). This generalization is stated here without proof; see Horn & Johnson §4.2 for the full treatment.

### Proof of the Maximum

Without loss of generality, restrict to $\|\mathbf{u}\| = 1$ (by scale-invariance). Using the principal axes form:

$$R(\mathbf{u}) = \mathbf{u}^T S \mathbf{u} = \sum_{i=1}^{n} \lambda_i \tilde{u}_i^2$$

subject to $\|\mathbf{u}\|^2 = 1$, which translates to $\sum_i \tilde{u}_i^2 = 1$ (since $Q$ is orthogonal).

We want to maximize $\sum_i \lambda_i \tilde{u}_i^2$ subject to $\sum_i \tilde{u}_i^2 = 1$, where all $\tilde{u}_i^2 \geq 0$. This is a convex combination of the $\lambda_i$, with weights $\tilde{u}_i^2 \geq 0$ summing to 1. A convex combination of numbers is at most the maximum of those numbers:

$$\sum_{i=1}^{n} \lambda_i \tilde{u}_i^2 \leq \lambda_{\max} \sum_{i=1}^{n} \tilde{u}_i^2 = \lambda_{\max}$$

Equality is achieved when all weight is on the index $k$ with $\lambda_k = \lambda_{\max}$: set $\tilde{u}_k = 1$ and $\tilde{u}_i = 0$ for $i \neq k$. In original coordinates, $\tilde{\mathbf{u}} = \mathbf{e}_k$ means $\mathbf{u} = Q \mathbf{e}_k = \mathbf{q}_k = \mathbf{q}_{\max}$.

The minimum follows by identical reasoning, replacing $\lambda_{\max}$ with $\lambda_{\min}$.

### So What? The PCA Connection

PCA asks: what unit vector $\mathbf{u}$ maximizes the variance of the data when projected onto $\mathbf{u}$? As derived in Section 10.1, this projected variance equals $\mathbf{u}^T C \mathbf{u}$, where $C = \frac{1}{n}X^T X$ is the sample covariance. This is exactly a Rayleigh quotient maximization:

$$\max_{\|\mathbf{u}\|=1} \mathbf{u}^T C \mathbf{u}$$

The Rayleigh-Ritz theorem tells us immediately: the answer is the leading eigenvector of $C$, with maximum variance equal to $\lambda_{\max}(C)$. The second principal component is the second eigenvector, and so on. The entire PCA algorithm is a direct consequence of the Rayleigh quotient result.

---

## 8. The Condition Number

### Definition

The **condition number** of a symmetric PD matrix $S$ is:

$$\kappa(S) = \frac{\lambda_{\max}}{\lambda_{\min}}$$

Since $S$ is PD, all eigenvalues are strictly positive, so $\kappa \geq 1$. A well-conditioned matrix has $\kappa$ close to 1; an ill-conditioned matrix has $\kappa \gg 1$.

*Note: the condition number is sometimes defined more generally as $\|S\| \cdot \|S^{-1}\|$ for arbitrary matrix norms, but for symmetric PD matrices with the spectral norm, this reduces to $\lambda_{\max}/\lambda_{\min}$. We use this specialized form throughout.*

### Geometric Meaning

Recall from Section 5 that the unit-level ellipsoid $\mathbf{u}^T S \mathbf{u} = 1$ has semi-axis lengths $1/\sqrt{\lambda_i}$. The ratio of the longest axis to the shortest axis is:

$$\frac{1/\sqrt{\lambda_{\min}}}{1/\sqrt{\lambda_{\max}}} = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}} = \sqrt{\kappa}$$

So $\sqrt{\kappa}$ is the **aspect ratio** of the ellipsoid. A sphere has $\kappa = 1$ (all axes equal). A thin needle-shaped ellipsoid has $\kappa \gg 1$.

### The Condition Number and Gradient Descent Convergence

The condition number directly controls how hard it is to minimize a quadratic loss by gradient descent. Here we derive the optimal convergence factor; the full derivation of the rate bound $\|x_t\| \leq \rho^t \|x_0\|$ it feeds into is in the companion notes *Gradient Descent on Quadratics: Convergence Analysis*.

In the eigenbasis, GD on a quadratic decouples into independent 1D processes: each eigendirection $i$ shrinks by a factor $(1 - \eta\lambda_i)$ per step. The overall per-step contraction factor is therefore the worst across all directions:

$$\rho(\eta) = \max_i |1 - \eta\lambda_i|$$

Since the eigenvalues all lie in the interval $[\lambda_{\min}, \lambda_{\max}]$ and $g(\lambda) = |1-\eta\lambda|$ is a V-shaped function of $\lambda$, its maximum over the interval is achieved at one of the two endpoints:

$$\rho(\eta) = \max\{|1 - \eta\lambda_{\min}|,\; |1 - \eta\lambda_{\max}|\}$$

To minimize $\rho$ over $\eta$, we want to **balance** these two endpoint values. Notice that as $\eta$ increases from 0, $|1-\eta\lambda_{\min}|$ decreases while $|1-\eta\lambda_{\max}|$ eventually increases past 1. The optimal balancing point must occur in the regime where $1/\lambda_{\max} < \eta < 1/\lambda_{\min}$: for $\eta < 1/\lambda_{\max}$ both terms are positive and $\lambda_{\max}$ still dominates (so we should increase $\eta$); for $\eta > 1/\lambda_{\min}$ both terms are negative and $\lambda_{\max}$ again dominates. Only in between does $1-\eta\lambda_{\min} > 0$ and $1-\eta\lambda_{\max} < 0$, so the balance equation becomes:

$$1 - \eta\lambda_{\min} = -(1 - \eta\lambda_{\max})$$

$$1 - \eta\lambda_{\min} = -1 + \eta\lambda_{\max}$$

$$2 = \eta(\lambda_{\min} + \lambda_{\max})$$

$$\eta^* = \frac{2}{\lambda_{\min} + \lambda_{\max}}$$

Plugging back in to find $\rho^*$:

$$\rho^* = |1 - \eta^* \lambda_{\max}| = \left|1 - \frac{2\lambda_{\max}}{\lambda_{\min} + \lambda_{\max}}\right| = \left|\frac{\lambda_{\min} + \lambda_{\max} - 2\lambda_{\max}}{\lambda_{\min} + \lambda_{\max}}\right| = \frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}}$$

Dividing numerator and denominator by $\lambda_{\min}$ and writing $\kappa = \lambda_{\max}/\lambda_{\min}$:

$$\boxed{\rho^*_{\text{GD}} = \frac{\kappa - 1}{\kappa + 1}}$$

As $\kappa \to 1$: $\rho^* \to 0$. Converges in very few steps.
As $\kappa \to \infty$: $\rho^* \to 1$. Convergence rate approaches zero — almost no progress per step.

**A concrete example.** With $\kappa = 100$:
- $\rho = 99/101 \approx 0.980$. After 100 steps: $0.98^{100} \approx 13\%$ of initial error remains.
- After 1000 steps: $0.98^{1000} \approx 2 \times 10^{-9}$ of initial error remains — finally converged.

With $\kappa = 10000$ the problem is far worse: you need $O(\kappa)$ steps to converge.

### Why Does High $\kappa$ Cause Slow Convergence?

The key tension: gradient descent uses a single learning rate $\alpha$ for all directions. The step size must be small enough to avoid overshooting the *steepest* direction (requires $\alpha < 2/\lambda_{\max}$), but this makes progress in the *shallowest* direction (where step size needed is $O(1/\lambda_{\min})$) glacially slow. The ratio of these two requirements is exactly $\kappa$.

This is why preconditioning, Newton's method, and momentum all exist — they each attempt to equalize the effective curvature across directions, reducing the effective condition number. These are developed fully in the companion notes *Gradient Descent on Quadratics: Convergence Analysis*.

So far we have only dealt with *pure* quadratic forms $\mathbf{u}^T S \mathbf{u}$. In practice, ML objectives often include a linear term as well. The next section shows how to handle this with a clean algebraic technique.

---

## 9. Completing the Square for Quadratics with Linear Terms

### Why This Matters

In ML, quadratic forms rarely appear in the pure form $\mathbf{u}^T S \mathbf{u}$ alone. More typically you encounter:

$$f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T S \mathbf{x} - \mathbf{b}^T \mathbf{x} + c$$

for some vector $\mathbf{b}$ and scalar $c$ — a quadratic with a linear term. This arises in least squares, ridge regression, Gaussian log-likelihoods, and quadratic approximations to arbitrary losses.

The technique for handling this is **completing the square** — the multivariate version of the scalar algebraic identity $(x - h)^2 = x^2 - 2hx + h^2$.

### Derivation

We want to rewrite $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T S \mathbf{x} - \mathbf{b}^T \mathbf{x} + c$ in the form $\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T S (\mathbf{x} - \boldsymbol{\mu}) + \text{const}$ for some $\boldsymbol{\mu}$.

Expand the candidate form:

$$\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T S (\mathbf{x} - \boldsymbol{\mu}) = \frac{1}{2} \mathbf{x}^T S \mathbf{x} - \boldsymbol{\mu}^T S \mathbf{x} + \frac{1}{2} \boldsymbol{\mu}^T S \boldsymbol{\mu}$$

(Here we used symmetry of $S$ to combine the two cross-terms: $\mathbf{x}^T S \boldsymbol{\mu} + \boldsymbol{\mu}^T S \mathbf{x} = 2\boldsymbol{\mu}^T S \mathbf{x}$.)

Matching the linear term against $-\mathbf{b}^T \mathbf{x}$:

$$-\boldsymbol{\mu}^T S \mathbf{x} = -\mathbf{b}^T \mathbf{x} \quad \Longrightarrow \quad S \boldsymbol{\mu} = \mathbf{b} \quad \Longrightarrow \quad \boldsymbol{\mu} = S^{-1} \mathbf{b}$$

(Assuming $S$ is PD and hence invertible.) Therefore:

$$\boxed{f(\mathbf{x}) = \frac{1}{2}(\mathbf{x} - S^{-1}\mathbf{b})^T S (\mathbf{x} - S^{-1}\mathbf{b}) + \left(c - \frac{1}{2} \mathbf{b}^T S^{-1} \mathbf{b}\right)}$$

The minimum is at $\mathbf{x}^* = S^{-1} \mathbf{b}$, with minimum value $c - \frac{1}{2} \mathbf{b}^T S^{-1} \mathbf{b}$.

### Verification by Gradient

As a sanity check: $\nabla_\mathbf{x} f = S\mathbf{x} - \mathbf{b} = \mathbf{0}$ gives $\mathbf{x}^* = S^{-1}\mathbf{b}$. ✓ Completing the square just expresses this geometrically: the minimum is at $S^{-1}\mathbf{b}$, and the function is a "bowl" of shape $S$ centered there.

With the core machinery in place — definition, symmetry, geometry, definiteness, eigendecomposition, sandwich inequality, Rayleigh quotient, condition number, and completing the square — we now survey where each of these tools shows up in practice.

---

## 10. Applications in ML

This section collects the recurring places where the quadratic form $\mathbf{u}^T S \mathbf{u}$ plays a starring role, with pointers to the relevant results above.

### 10.1 The Sample Covariance Matrix as a Quadratic Form

#### Setup and Motivation

The covariance matrix is one of the most important symmetric PSD matrices in ML, and it appears in PCA, Gaussian models, Mahalanobis distance, and linear discriminant analysis. Before using it in those contexts, it's worth deriving it from scratch so its quadratic form structure is transparent.

Let $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^d$ be $n$ data points, each a $d$-dimensional observation. Stack them as rows into a data matrix:

$$X = \begin{pmatrix} — \mathbf{x}_1^T — \\ \vdots \\ — \mathbf{x}_n^T — \end{pmatrix} \in \mathbb{R}^{n \times d}$$

**Centering assumption:** We assume the data is mean-zero, i.e., $\frac{1}{n}\sum_{i=1}^n \mathbf{x}_i = \mathbf{0}$. In practice you subtract the empirical mean first; assuming it's already done keeps the notation clean.

#### Deriving the Sample Covariance Matrix

Recall that the covariance between two scalar random variables $a$ and $b$ (empirically, over $n$ samples) is $\frac{1}{n}\sum_{i=1}^n a_i b_i$ (using mean-zero data). For two coordinates $j$ and $k$ of our data, the empirical covariance is:

$$C_{jk} = \frac{1}{n} \sum_{i=1}^n x_{ij} \, x_{ik}$$

where $x_{ij}$ denotes the $j$-th coordinate of observation $\mathbf{x}_i$. Recognizing this as a dot product between the $j$-th and $k$-th *columns* of $X$, we can write the entire $d \times d$ covariance matrix compactly as:

$$C = \frac{1}{n} X^T X$$

To verify: the $(j,k)$ entry of $X^T X$ is the dot product of column $j$ of $X$ with column $k$ of $X$, which is exactly $\sum_{i=1}^n x_{ij} x_{ik}$. Dividing by $n$ gives $C_{jk}$ as defined above. ✓

*Convention note:* Some sources divide by $n-1$ (Bessel's correction) for an unbiased estimate of the population covariance. Here we use $1/n$ (the MLE of the covariance of a Gaussian). The distinction doesn't affect the quadratic form structure or any of the following analysis.

#### $C$ is Positive Semidefinite

**Claim:** $C = \frac{1}{n} X^T X$ is PSD — i.e., $\mathbf{u}^T C \mathbf{u} \geq 0$ for all $\mathbf{u} \in \mathbb{R}^d$.

**Proof:** Substitute directly:

$$\mathbf{u}^T C \mathbf{u} = \mathbf{u}^T \frac{1}{n} X^T X \mathbf{u} = \frac{1}{n} (X\mathbf{u})^T (X\mathbf{u}) = \frac{1}{n} \|X\mathbf{u}\|^2 \geq 0$$

The step $\mathbf{u}^T X^T X \mathbf{u} = (X\mathbf{u})^T(X\mathbf{u})$ uses the rule $(AB)^T = B^T A^T$: $(X\mathbf{u})^T = \mathbf{u}^T X^T$. A squared norm is always non-negative, so the whole expression is $\geq 0$. ✓

**When is $C$ positive definite (strictly $> 0$)?** Only when $X\mathbf{u} \neq \mathbf{0}$ for all nonzero $\mathbf{u}$, i.e., when $X$ has full column rank ($\text{rank}(X) = d$). In practice:

- If $n < d$ (fewer observations than features), $X$ cannot have rank $d$, so $C$ is singular (PSD but not PD). PCA still works in this case but $C$ has zero eigenvalues.
- If $n \geq d$ and the data has no exactly collinear features, $C$ is PD.

#### $\mathbf{u}^T C \mathbf{u}$ Is the Projected Variance

The quadratic form $\mathbf{u}^T C \mathbf{u}$ has a direct statistical interpretation. Consider projecting each data point onto the direction $\mathbf{u}$: the scalar projection of $\mathbf{x}_i$ is $z_i = \mathbf{u}^T \mathbf{x}_i$. The empirical variance of these projected scalars (using the mean-zero assumption) is:

$$\frac{1}{n}\sum_{i=1}^n z_i^2 = \frac{1}{n}\sum_{i=1}^n (\mathbf{u}^T \mathbf{x}_i)^2 = \frac{1}{n} \|X\mathbf{u}\|^2 = \mathbf{u}^T C \mathbf{u}$$

So $\mathbf{u}^T C \mathbf{u}$ is the variance of the data *when you look at it from direction* $\mathbf{u}$. Changing $\mathbf{u}$ rotates your viewing angle; the quadratic form tells you how spread out the data appears from each angle.

This interpretation is worth pausing on. The covariance matrix $C$ doesn't just store pairwise covariances — it *encodes the variance of the data in every possible direction simultaneously*, via the quadratic form. The eigenvectors of $C$ are the directions of maximum, second-maximum, etc. variance; the eigenvalues are the corresponding variances. This is precisely why the Rayleigh quotient (Section 7) solves PCA (Section 10.3).

### 10.2 Local Quadratic Approximation of a Loss

Any twice-differentiable loss $L(\boldsymbol{\theta})$, Taylor-expanded around a point $\boldsymbol{\theta}_0$, has a local quadratic form in its second-order term:

$$L(\boldsymbol{\theta}_0 + \Delta\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}_0) + \nabla L(\boldsymbol{\theta}_0)^T \Delta\boldsymbol{\theta} + \frac{1}{2} \Delta\boldsymbol{\theta}^T H(\boldsymbol{\theta}_0) \Delta\boldsymbol{\theta}$$

Here $H(\boldsymbol{\theta}_0)$ is the Hessian (symmetric by Clairaut's theorem for smooth $L$). Near a local minimum where $\nabla L = \mathbf{0}$, the loss surface is approximately $\frac{1}{2} \Delta\boldsymbol{\theta}^T H \Delta\boldsymbol{\theta}$ — a pure quadratic form. The eigenvalues of $H$ determine the shape of this local bowl (Section 5), the condition number determines how hard it is to optimize (Section 8), and the sandwich inequality (Section 6) underpins the convergence proofs.

### 10.3 PCA: Variance as a Rayleigh Quotient

The sample covariance matrix is derived in Section 10.1. The key result there is:

$$\mathbf{u}^T C \mathbf{u} = \text{empirical variance of } \{\mathbf{u}^T \mathbf{x}_i\}$$

PCA asks: which unit direction $\mathbf{u}$ maximizes this projected variance? This is exactly a Rayleigh quotient maximization (Section 7):

$$\max_{\|\mathbf{u}\|=1} \mathbf{u}^T C \mathbf{u}$$

The Rayleigh-Ritz theorem gives the answer immediately: the maximizing direction is the leading eigenvector of $C$, with maximum variance $\lambda_{\max}(C)$. The second principal component is the direction of maximum variance among all vectors orthogonal to the first, which is the second eigenvector of $C$ — and so on. The entire PCA algorithm is a direct consequence of the quadratic form structure of variance and the Rayleigh-Ritz theorem.

### 10.4 Ridge Regression

Ordinary least squares minimizes $\|X\boldsymbol{\theta} - \mathbf{y}\|^2 = \boldsymbol{\theta}^T X^T X \boldsymbol{\theta} - 2\mathbf{y}^T X \boldsymbol{\theta} + \|\mathbf{y}\|^2$. This is a quadratic in $\boldsymbol{\theta}$ with matrix $S = X^T X$.

When $X^T X$ is singular or near-singular (ill-conditioned), the solution is numerically unstable because $\kappa(X^T X)$ is huge (Section 8). Ridge regression adds $\lambda \|\boldsymbol{\theta}\|^2 = \lambda \boldsymbol{\theta}^T I \boldsymbol{\theta}$ to the loss, which modifies the matrix to $X^T X + \lambda I$. Every eigenvalue shifts by $+\lambda$:

$$\lambda_i(X^T X + \lambda I) = \lambda_i(X^T X) + \lambda$$

**Proof:** If $\mathbf{q}$ is an eigenvector of $X^TX$ with eigenvalue $\mu$, then $(X^TX + \lambda I)\mathbf{q} = X^TX\mathbf{q} + \lambda\mathbf{q} = \mu\mathbf{q} + \lambda\mathbf{q} = (\mu + \lambda)\mathbf{q}$. So $\mathbf{q}$ is also an eigenvector of $X^TX + \lambda I$, with eigenvalue $\mu + \lambda$. Since this holds for every eigenvector of $X^TX$, every eigenvalue shifts by exactly $\lambda$. ✓

Since all eigenvalues increase by $\lambda$, the minimum eigenvalue is now at least $\lambda > 0$ (even if $X^T X$ had zero eigenvalues). The condition number becomes:

$$\kappa(X^T X + \lambda I) = \frac{\lambda_{\max}(X^T X) + \lambda}{\lambda_{\min}(X^T X) + \lambda}$$

As $\lambda \to \infty$, $\kappa \to 1$ — the problem becomes perfectly conditioned (but we're adding large bias). Choosing $\lambda$ trades off between conditioning (lower $\kappa$ = easier optimization, more stable solution) and bias (large $\lambda$ shrinks the solution toward zero).

### 10.5 Mahalanobis Distance

The **Mahalanobis distance** between a point $\mathbf{x}$ and a distribution with mean $\boldsymbol{\mu}$ and covariance $\Sigma$ is:

$$d_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

This is the square root of a quadratic form in $(\mathbf{x} - \boldsymbol{\mu})$ with matrix $\Sigma^{-1}$. The covariance matrix $\Sigma$ here plays the same role as $C$ in Section 10.1 (population version instead of sample version). Using $\Sigma^{-1}$ rather than $\Sigma$ inverts the relationship between eigenvalues and axis lengths: directions of high variance (large eigenvalues of $\Sigma$, small eigenvalues of $\Sigma^{-1}$) contribute *less* to distance, and directions of low variance contribute *more*. Intuitively, a point that is far from the mean along a high-variance axis is unremarkable; a point that is far along a low-variance axis is genuinely unusual.

Geometrically, the level sets $d_M(\mathbf{x}) = c$ are ellipsoids aligned with the axes of $\Sigma$ — exactly the structure from Section 5. The log-likelihood of a Gaussian is:

$$\log p(\mathbf{x}) = -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) + \text{const}$$

So maximizing Gaussian log-likelihood is the same as minimizing the Mahalanobis distance squared — a direct quadratic form optimization.

### 10.6 Gradient Descent Analysis: The L-Smoothness Descent Lemma

The **descent lemma** is the key result that makes gradient descent convergence proofs work. It says: if $L$ is $L$-smooth — meaning $\lambda_{\max}(H(\boldsymbol{\theta})) \leq L$ everywhere, equivalently $H \preceq L \cdot I$ — then a gradient step with $\alpha \leq 1/L$ is guaranteed to decrease the loss. Here we derive it explicitly.

**Setup.** The $L$-smoothness condition $H \preceq L \cdot I$ means $\mathbf{v}^T H \mathbf{v} \leq L\|\mathbf{v}\|^2$ for all $\mathbf{v}$ (by the sandwich inequality, Section 6). This implies the following quadratic upper bound on $L$: for any $\boldsymbol{\theta}$ and displacement $\mathbf{d}$,

$$L(\boldsymbol{\theta} + \mathbf{d}) \leq L(\boldsymbol{\theta}) + \nabla L(\boldsymbol{\theta})^T \mathbf{d} + \frac{L}{2}\|\mathbf{d}\|^2$$

This is the Taylor expansion with the second-order term replaced by its worst-case upper bound $\frac{L}{2}\|\mathbf{d}\|^2$ (using $\frac{1}{2}\mathbf{d}^T H \mathbf{d} \leq \frac{L}{2}\|\mathbf{d}\|^2$ from the sandwich). The proof that this bound holds for all $\boldsymbol{\theta}$, not just locally, requires integrating the Hessian bound along the line segment — that argument is standard but out of scope here; see Boyd & Vandenberghe §9.1.

**Deriving the descent guarantee.** Now set $\mathbf{d} = -\alpha \nabla L(\boldsymbol{\theta})$ (a single gradient step). Substitute into the upper bound:

$$L(\boldsymbol{\theta} - \alpha \nabla L) \leq L(\boldsymbol{\theta}) + \nabla L^T(-\alpha \nabla L) + \frac{L}{2}\|\alpha \nabla L\|^2$$

Simplify each term. The inner product: $\nabla L^T(-\alpha \nabla L) = -\alpha\|\nabla L\|^2$. The squared norm: $\frac{L}{2}\alpha^2\|\nabla L\|^2$. Combining:

$$L(\boldsymbol{\theta} - \alpha \nabla L) \leq L(\boldsymbol{\theta}) - \alpha\|\nabla L\|^2 + \frac{L\alpha^2}{2}\|\nabla L\|^2 = L(\boldsymbol{\theta}) - \alpha\left(1 - \frac{\alpha L}{2}\right)\|\nabla L(\boldsymbol{\theta})\|^2$$

$$\boxed{L(\boldsymbol{\theta} - \alpha \nabla L) \leq L(\boldsymbol{\theta}) - \alpha\left(1 - \frac{\alpha L}{2}\right) \|\nabla L(\boldsymbol{\theta})\|^2}$$

The decrease is guaranteed when the coefficient of $\|\nabla L\|^2$ is positive: $1 - \frac{\alpha L}{2} > 0$, i.e., $\alpha < 2/L$.

**So what?** The amount of decrease per step is $\alpha(1 - \frac{\alpha L}{2})\|\nabla L\|^2$. This is maximised over $\alpha$ by setting the derivative to zero: $\frac{d}{d\alpha}[\alpha - \frac{\alpha^2 L}{2}] = 1 - \alpha L = 0$, giving $\alpha^* = 1/L$. At this step size the per-step decrease is $\frac{1}{2L}\|\nabla L\|^2$. The connection to the sandwich inequality: $L = \lambda_{\max}(H)$, so choosing $\alpha = 1/L = 1/\lambda_{\max}$ is exactly the stability condition $\eta < 2/\lambda_{\max}$ from the companion notes, at its optimal value.

---

## 11. Sources and Further Reading

Quadratic forms are standard linear algebra; the treatment here synthesizes classical results without a single dominant source.

- **Strang, *Introduction to Linear Algebra***, Ch. 6–7. Accessible development of the spectral theorem and positive definiteness with strong geometric intuition.
- **Horn & Johnson, *Matrix Analysis***, Ch. 4, 7. Definitive reference for the spectral theorem, Rayleigh quotient, and variational characterizations of eigenvalues. Use for rigorous proofs.
- **Boyd & Vandenberghe, *Convex Optimization***, Appendix A.1, and Ch. 9. The condition number and L-smoothness/strong convexity conditions in the optimization context; the descent lemma appears in §9.1.
- **Nocedal & Wright, *Numerical Optimization***, Ch. 2. Condition number and its role in convergence rates of gradient descent and Newton methods.
- For the **PCA as Rayleigh quotient** framing, see any standard ML textbook (e.g., Bishop's *PRML*, Ch. 12) — this is a standard result, not cleanly attributable to a single source.
- The **ridge regression eigenvalue shift** interpretation is standard in regularization literature; see e.g., Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, §3.4.
