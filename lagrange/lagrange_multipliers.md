# Lagrange Multipliers and Constrained Optimization

---

## Table of Contents

1. [Motivation: Why Constrained Optimization Is Different](#1-motivation-why-constrained-optimization-is-different)
2. [Equality Constraints: The Lagrange Multiplier Method](#2-equality-constraints-the-lagrange-multiplier-method)
   - 2.1 Geometric Intuition
   - 2.2 The Lagrangian
   - 2.3 First-Order Conditions
   - 2.4 Multiple Equality Constraints
   - 2.5 The Shadow Price Interpretation
   - 2.6 Worked Example
3. [Second-Order Conditions](#3-second-order-conditions)
   - 3.1 Why First-Order Conditions Aren't Enough
   - 3.2 The Tangent Space
   - 3.3 The Constrained Hessian Condition
   - 3.4 The Bordered Hessian
   - 3.5 Worked Example Continued
4. [Inequality Constraints: The KKT Conditions](#4-inequality-constraints-the-kkt-conditions)
   - 4.1 The New Difficulty
   - 4.2 The KKT Lagrangian
   - 4.3 The Four KKT Conditions
   - 4.4 Unpacking Each Condition
   - 4.5 Constraint Qualifications
   - 4.6 Necessity vs. Sufficiency
   - 4.7 Worked Example
5. [Lagrangian Duality](#5-lagrangian-duality)
   - 5.1 The Dual Function
   - 5.2 Weak Duality
   - 5.3 Strong Duality and Slater's Condition
   - 5.4 KKT as Primal-Dual Optimality
6. [ML Applications](#6-ml-applications)
   - 6.1 Regularization as a Constraint
   - 6.2 Maximum Entropy and the Exponential Family
7. [Summary and Quick Reference](#7-summary-and-quick-reference)
8. [Sources and Further Reading](#8-sources-and-further-reading)

---

## 1. Motivation: Why Constrained Optimization Is Different

In unconstrained optimization, a necessary condition for $x^*$ to be a local minimum of $f : \mathbb{R}^n \to \mathbb{R}$ is:

$$\nabla f(x^*) = 0$$

The intuition is immediate: if $\nabla f(x^*) \neq 0$, then taking a small step in direction $-\nabla f(x^*)$ decreases $f$, so $x^*$ cannot be a minimum.

Now add a constraint. The feasible set is no longer all of $\mathbb{R}^n$ but some subset defined by $g(x) = 0$ or $g(x) \leq 0$. The gradient condition breaks down entirely: **even when $\nabla f(x^*) \neq 0$, the point $x^*$ can be a constrained minimum**, because moving in direction $-\nabla f(x^*)$ might leave the feasible set.

**A concrete example.** Let $f(x, y) = x + y$ and constrain to the unit circle $g(x,y) = x^2 + y^2 - 1 = 0$. The unconstrained minimum of $f$ doesn't even exist ($f \to -\infty$). The constrained minimum on the circle is at $x^* = y^* = -1/\sqrt{2}$, giving $f = -\sqrt{2}$. At this point, $\nabla f = (1,1) \neq 0$ — the unconstrained condition fails completely.

**The naïve approach and why it fails.** One idea: substitute the constraint to eliminate a variable, then minimize the resulting unconstrained problem. For $g(x,y) = 0$ you could write $y = y(x)$ and minimize $f(x, y(x))$ over $x$ alone.

This has severe limitations:
- Constraints are often hard to invert analytically (e.g., $g(x,y) = x^3 + y^3 - 1 = 0$).
- With $m$ equality constraints you eliminate $m$ variables, destroying the problem's symmetry.
- It gives no clean generalization to inequality constraints or to multiple constraints.
- It offers no geometric insight and no connection to the dual problem.

Lagrange's method gives a unified, systematic treatment that avoids all these issues. It will be one of the most frequently used tools in your ML career.

---

## 2. Equality Constraints: The Lagrange Multiplier Method

### 2.1 Geometric Intuition

**Setup.** Minimize $f(x)$ subject to $g(x) = 0$, where $x \in \mathbb{R}^n$, $f : \mathbb{R}^n \to \mathbb{R}$, and $g : \mathbb{R}^n \to \mathbb{R}$.

The constraint $g(x) = 0$ defines an $(n-1)$-dimensional surface in $\mathbb{R}^n$ — a *manifold* (smooth curved subset). Think of it as a curved $(n-1)$-dimensional "wall" that $x$ must stay on.

**Fact: $\nabla g$ is orthogonal to the constraint surface.** Here's why. Take any smooth curve $x(t)$ lying on the surface, so $g(x(t)) = 0$ for all $t$. Differentiate both sides with respect to $t$ using the chain rule:

$$\frac{d}{dt} g(x(t)) = \nabla g(x(t))^T \dot{x}(t) = 0$$

So $\nabla g(x(t))$ is orthogonal to the velocity $\dot{x}(t)$ of every curve on the surface. Since this holds for every such curve, $\nabla g$ is orthogonal to every direction tangent to the surface — i.e., $\nabla g$ is a *normal* to the surface.

**The optimality argument.** Suppose $x^*$ is a constrained minimum of $f$ on the surface $g(x) = 0$. Take any smooth curve $x(t)$ on the surface with $x(0) = x^*$. Since $x^*$ is a minimum, $\varphi(t) := f(x(t))$ achieves its minimum at $t=0$, so $\varphi'(0) = 0$:

$$\varphi'(0) = \nabla f(x^*)^T \dot{x}(0) = 0$$

This must hold for **every** tangent direction $\dot{x}(0)$ to the surface at $x^*$. In other words: $\nabla f(x^*)$ is orthogonal to every tangent direction.

We just showed $\nabla g(x^*)$ is also orthogonal to every tangent direction. The tangent space has dimension $n-1$, so its orthogonal complement — the normal space — has dimension 1. Two nonzero vectors orthogonal to the same $(n-1)$-dimensional subspace of $\mathbb{R}^n$ must be parallel. Therefore:

$$\boxed{\nabla f(x^*) = \lambda \, \nabla g(x^*)}$$

for some scalar $\lambda \in \mathbb{R}$. This scalar is the **Lagrange multiplier**.

**Plain-language reading:** At a constrained optimum, $\nabla f$ and $\nabla g$ point in the same direction (up to sign and scaling). If they didn't — if $\nabla f$ had any component *along* the surface — you could move along the surface in the direction that decreases $f$, contradicting optimality. So all of $\nabla f$'s "push" must be perpendicular to the surface, and that perpendicular direction is exactly $\nabla g$.

**Quick verification.** For $f(x,y) = x + y$ on the unit circle $g(x,y) = x^2 + y^2 - 1 = 0$: $\nabla f = (1,1)$, $\nabla g = (2x, 2y)$. Setting $\nabla f = \lambda \nabla g$:

$$1 = 2\lambda x, \quad 1 = 2\lambda y \implies x = y = \frac{1}{2\lambda}$$

Substituting into the constraint: $2/(4\lambda^2) = 1 \implies \lambda = \pm 1/\sqrt{2}$. This gives two critical points:

- $\lambda = -1/\sqrt{2}$: $x^* = y^* = -1/\sqrt{2}$ (the minimum, $f = -\sqrt{2}$) ✓
- $\lambda = +1/\sqrt{2}$: $x^* = y^* = +1/\sqrt{2}$ (the maximum, $f = +\sqrt{2}$)

The method finds both — distinguishing them requires second-order conditions (Section 3).

### 2.2 The Lagrangian

The geometric argument gives us the optimality condition $\nabla f = \lambda \nabla g$. We now package this into a single function that makes computation systematic.

**Definition (The Lagrangian).** The *Lagrangian* for the equality-constrained problem is:

$$\mathcal{L}(x, \lambda) = f(x) - \lambda \, g(x)$$

where $\lambda \in \mathbb{R}$ is the *Lagrange multiplier* (also called the *dual variable*).

**Sign convention note.** The minus sign in $-\lambda g(x)$ is a convention. Since $\lambda$ is a free variable (any real number, positive or negative), writing $+\lambda g(x)$ or $-\lambda g(x)$ produces the same set of solutions — a sign flip in $\lambda$ accounts for the difference. We use minus here to match a common convention in calculus and economics; Boyd & Vandenberghe use plus. Be consistent within any given derivation.

**Why is this useful?** Because setting the gradient of $\mathcal{L}$ to zero gives exactly the conditions we want:

$$\nabla_x \mathcal{L} = \nabla f(x) - \lambda \nabla g(x) = 0 \quad \Longleftrightarrow \quad \nabla f(x) = \lambda \nabla g(x) \quad \text{(gradient alignment)}$$

$$\frac{\partial \mathcal{L}}{\partial \lambda} = -g(x) = 0 \quad \Longleftrightarrow \quad g(x) = 0 \quad \text{(feasibility)}$$

The Lagrangian trades a *constrained $n$-dimensional* optimization problem for an *unconstrained $(n+1)$-dimensional* problem in the variables $(x_1, \ldots, x_n, \lambda)$. The constraint is automatically enforced by differentiating with respect to $\lambda$.

### 2.3 First-Order Conditions

Setting $\nabla_{x,\lambda} \mathcal{L} = 0$ gives the **Lagrange (first-order) conditions**:

$$\underbrace{\nabla_x \mathcal{L} = 0}_{\text{stationarity}} : \quad \nabla f(x^*) = \lambda \nabla g(x^*)$$

$$\underbrace{\frac{\partial \mathcal{L}}{\partial \lambda} = 0}_{\text{feasibility}} : \quad g(x^*) = 0$$

This is $n + 1$ equations in $n + 1$ unknowns $(x_1, \ldots, x_n, \lambda)$.

**Important caveat — necessity, not sufficiency.** These conditions identify *stationary points* of the Lagrangian — points where no first-order feasible move changes $f$. A stationary point might be a minimum, maximum, or saddle point. Second-order conditions (Section 3) are needed to classify it.

**Another caveat — constraint qualifications.** The geometric argument assumed $\nabla g(x^*) \neq 0$. If $\nabla g(x^*) = 0$ at the optimal point, the normal direction is not well-defined, and the argument breaks down. The condition $\nabla g(x^*) \neq 0$ is the simplest example of a *constraint qualification* (CQ). We'll treat CQs carefully in Section 4.5 for the inequality case; for now, assume $\nabla g(x^*) \neq 0$.

### 2.4 Multiple Equality Constraints

**Notation note.** From here onward we adopt the convention used in §4 and throughout ML: $h_i$ denotes equality constraint functions and $g_j$ denotes inequality constraint functions. The single equality constraint in §2.1–2.3 was called $g$ for simplicity; that $g$ maps to what we now call $h$.

**Setup.** Minimize $f(x)$ subject to $h_i(x) = 0$ for $i = 1, \ldots, m$, where $x \in \mathbb{R}^n$ and $m < n$.

With $m$ constraints, the feasible surface has dimension $n - m$. At each point $x^*$ on it, $\nabla h_1(x^*), \ldots, \nabla h_m(x^*)$ are each normal to the surface. If these gradients are *linearly independent* — meaning no one of them is a linear combination of the others — they span the entire $m$-dimensional normal space. (This is the **LICQ** condition, defined precisely in Section 4.5.)

By the same geometric argument: $\nabla f(x^*)$ must lie in the normal space at $x^*$, so it must be a linear combination of the constraint gradients:

$$\nabla f(x^*) = \sum_{i=1}^m \lambda_i \nabla h_i(x^*)$$

**Lagrangian for multiple equality constraints:**

$$\mathcal{L}(x, \lambda) = f(x) - \sum_{i=1}^m \lambda_i h_i(x)$$

where $\lambda = (\lambda_1, \ldots, \lambda_m) \in \mathbb{R}^m$.

**First-order conditions:**

$$\nabla_x \mathcal{L} = \nabla f(x) - \sum_{i=1}^m \lambda_i \nabla h_i(x) = 0 \quad \text{(stationarity)}$$

$$h_i(x) = 0 \quad \text{for all } i \quad \text{(feasibility)}$$

This gives $n + m$ equations in $n + m$ unknowns $(x, \lambda)$.

### 2.5 The Shadow Price Interpretation

The Lagrange multiplier has a beautiful economic interpretation that is important throughout optimization and ML.

**Setup.** Consider the parameterized problem:

$$p^*(c) = \min_{x} f(x) \quad \text{subject to} \quad g(x) = c$$

where $c \in \mathbb{R}$ is a parameter that controls where the constraint is set. The quantity $p^*(c)$ is the *optimal value function* — it tells us the best achievable value of $f$ as a function of the constraint level $c$.

**Claim:** $\dfrac{dp^*}{dc} = \lambda^*$, where $\lambda^*$ is the Lagrange multiplier at the solution $x^*(c)$.

**Derivation.** We have two facts. First, from the constraint: $g(x^*(c)) = c$ for all $c$. Differentiating both sides with respect to $c$:

$$\nabla g(x^*)^T \frac{dx^*}{dc} = 1 \quad \quad (*)$$

Second, $p^*(c) = f(x^*(c))$. Differentiating with respect to $c$ using the chain rule:

$$\frac{dp^*}{dc} = \nabla f(x^*)^T \frac{dx^*}{dc}$$

From the stationarity condition, $\nabla f(x^*) = \lambda^* \nabla g(x^*)$. Substituting:

$$\frac{dp^*}{dc} = \lambda^* \nabla g(x^*)^T \frac{dx^*}{dc} = \lambda^* \cdot 1 = \lambda^*$$

where the last step used $(*)$.

**So what?** The Lagrange multiplier $\lambda^*$ is the rate at which the optimal value changes when the constraint is relaxed or tightened by a small amount. In economics this is called the *shadow price* of the constraint: it measures how much you'd be willing to "pay" (in units of the objective) to loosen the constraint by one unit.

**Concrete example.** In Section 2.6 we'll find $\lambda^* = b / (a^T Q^{-1} a)$ for the problem of minimizing $\frac{1}{2} x^T Q x$ on the hyperplane $a^T x = b$. The shadow price interpretation says: if we shift the constraint to $a^T x = b + \varepsilon$, the optimal cost changes by approximately $\lambda^* \varepsilon$. A larger $b$ (pushing the constraint further out) costs more, and $\lambda^*$ quantifies exactly how much.

This interpretation reappears directly in Section 6.1 (regularization parameter = shadow price of a norm constraint) and underlies the duality theory in Section 5.

### 2.6 Worked Example: Minimizing a Quadratic on a Hyperplane

**Problem.** Minimize $f(x) = \frac{1}{2} x^T Q x$ subject to $a^T x = b$, where $Q \in \mathbb{R}^{n \times n}$ is *symmetric positive definite* (*SPD*: symmetric means $Q = Q^T$; positive definite means $v^T Q v > 0$ for all $v \neq 0$, equivalently all eigenvalues are strictly positive), $a \in \mathbb{R}^n$, and $b \in \mathbb{R}$.

**Why SPD?** It ensures $f$ is strictly convex (the level sets are ellipsoids) and that $Q^{-1}$ exists, both of which we'll need.

Write the constraint as $h(x) = a^T x - b = 0$. Then $\nabla h = a$ (the gradient of a linear function $a^T x$ is just $a$).

**Lagrangian:**

$$\mathcal{L}(x, \lambda) = \frac{1}{2} x^T Q x - \lambda(a^T x - b)$$

**Stationarity ($\nabla_x \mathcal{L} = 0$):**

$$Qx - \lambda a = 0 \implies Qx = \lambda a \implies x = \lambda Q^{-1} a$$

Here we use that $Q$ is SPD (hence invertible) to left-multiply by $Q^{-1}$.

**Feasibility ($h(x) = 0$):**

$$a^T(\lambda Q^{-1} a) = b \implies \lambda (a^T Q^{-1} a) = b \implies \lambda^* = \frac{b}{a^T Q^{-1} a}$$

Note that $a^T Q^{-1} a > 0$ since $Q^{-1}$ is also SPD (the inverse of a PD matrix is PD), so the denominator is nonzero.

**Optimal solution:**

$$x^* = \frac{b}{a^T Q^{-1} a} \, Q^{-1} a$$

**Interpretation.** The vector $Q^{-1} a$ is the direction of "steepest descent toward the hyperplane" as measured by the quadratic metric $Q$. The scalar $b / (a^T Q^{-1} a)$ scales this direction to exactly hit the hyperplane $a^T x = b$.

The shadow price is $\lambda^* = b / (a^T Q^{-1} a)$: shifting the hyperplane to $a^T x = b + \varepsilon$ changes the optimal cost $\frac{1}{2} (x^*)^T Q x^*$ by approximately $\lambda^* \varepsilon$.

---

## 3. Second-Order Conditions

### 3.1 Why First-Order Conditions Aren't Enough

The first-order conditions identify stationary points of $\mathcal{L}$ — points where no first-order move changes $f$. But a stationary point could be a minimum, maximum, or saddle point, just as in unconstrained optimization.

In unconstrained optimization, we distinguish these using the Hessian $\nabla^2 f$:
- Positive definite Hessian $\Rightarrow$ local minimum
- Negative definite Hessian $\Rightarrow$ local maximum
- Indefinite Hessian $\Rightarrow$ saddle point

In constrained optimization, the analogous test applies to the Hessian of the *Lagrangian*, but only in the *directions tangent to the constraint surface* — because those are the only directions we can move while remaining feasible.

### 3.2 The Tangent Space

**Definition (Tangent space).** At a point $x^*$ satisfying $h(x^*) = 0$, the *tangent space* to the constraint surface is:

$$T_{x^*} = \{ d \in \mathbb{R}^n : \nabla h(x^*)^T d = 0 \}$$

*Reading this:* $T_{x^*}$ is the set of directions $d$ that are perpendicular to $\nabla h(x^*)$. Since $\nabla h(x^*)$ is normal to the surface, the directions perpendicular to it are exactly the directions tangent to the surface.

*Concrete example.* Constraint: unit circle $h(x,y) = x^2 + y^2 - 1 = 0$. At $x^* = (1,0)$: $\nabla h = (2, 0)$. Tangent space: $\{d : 2d_1 = 0\} = \{d : d_1 = 0\}$, i.e., the vertical direction $(0, t)$ for $t \in \mathbb{R}$. This is correct — at the rightmost point of the circle, you can only move up or down along the circle, not left or right.

For multiple constraints $h_i(x) = 0$, $i = 1, \ldots, m$:

$$T_{x^*} = \{ d \in \mathbb{R}^n : \nabla h_i(x^*)^T d = 0 \text{ for all } i = 1, \ldots, m \}$$

This is the intersection of all the individual tangent spaces. With $m$ independent constraints, $T_{x^*}$ has dimension $n - m$.

### 3.3 The Constrained Hessian Condition

At a constrained stationary point $(x^*, \lambda^*)$ satisfying the first-order conditions, define the **Hessian of the Lagrangian** with respect to $x$. Using the minus-convention Lagrangian from Section 2.2 ($\mathcal{L} = f - \sum \lambda_i h_i$):

$$\nabla^2_x \mathcal{L} = \nabla^2 f(x^*) - \sum_{i=1}^m \lambda_i^* \nabla^2 h_i(x^*)$$

**Sign convention note.** When using the plus-convention Lagrangian from Section 4.2 ($\mathcal{L} = f + \sum \lambda_i h_i + \sum \mu_j g_j$), this formula becomes $\nabla^2_x \mathcal{L} = \nabla^2 f + \sum \lambda_i^* \nabla^2 h_i + \sum \mu_j^* \nabla^2 g_j$. These two expressions agree numerically at the solution, because the $\lambda^*$ values produced by each convention differ in sign — but they are written differently. Regardless of convention, the Hessian of the Lagrangian is always just the matrix of second partial derivatives of $\mathcal{L}$ with respect to $x$, evaluated at $(x^*, \lambda^*)$ or $(x^*, \lambda^*, \mu^*)$.

This differs from $\nabla^2 f$ alone: the correction terms account for the curvature of the constraint surface. If the constraints are linear (e.g., $h_i(x) = a_i^T x - b_i$), then $\nabla^2 h_i = 0$ and $\nabla^2_x \mathcal{L} = \nabla^2 f$. For nonlinear constraints, the correction matters.

**Second-order necessary condition for local minimum:**

$$d^T \nabla^2_x \mathcal{L} \, d \geq 0 \quad \text{for all } d \in T_{x^*}$$

**Second-order sufficient condition for local minimum:**

$$d^T \nabla^2_x \mathcal{L} \, d > 0 \quad \text{for all } d \in T_{x^*}, \; d \neq 0$$

We say $\nabla^2_x \mathcal{L}$ is *positive definite on the tangent space* $T_{x^*}$.

*Note carefully:* We do **not** require $\nabla^2_x \mathcal{L}$ to be positive definite on all of $\mathbb{R}^n$ — only on the subspace of feasible directions. In directions perpendicular to the surface (i.e., in the direction of $\nabla h$), $\nabla^2_x \mathcal{L}$ can be anything.

Analogously, for a constrained local **maximum**: $d^T \nabla^2_x \mathcal{L} \, d < 0$ for all nonzero $d \in T_{x^*}$.

### 3.4 The Bordered Hessian

Checking "is $\nabla^2_x \mathcal{L}$ positive definite on $T_{x^*}$?" directly requires finding the tangent space, projecting onto it, and checking definiteness — algebraically tedious. The *bordered Hessian* provides an equivalent algebraic test using determinants.

**Definition (Bordered Hessian, one equality constraint).** For $h(x) = 0$, $x \in \mathbb{R}^n$, the bordered Hessian at $(x^*, \lambda^*)$ is the $(n+1) \times (n+1)$ matrix:

$$\bar{H} = \begin{pmatrix} 0 & \nabla h^T \\ \nabla h & \nabla^2_x \mathcal{L} \end{pmatrix}$$

Here: $\nabla h = \nabla h(x^*) \in \mathbb{R}^n$ is a column vector; $\nabla h^T$ is its transpose (a row vector); and $\nabla^2_x \mathcal{L} \in \mathbb{R}^{n \times n}$. The scalar $0$ in the top-left fills the $(1,1)$ entry. The full structure is:

$$\bar{H} = \begin{pmatrix} 0 & \frac{\partial h}{\partial x_1} & \frac{\partial h}{\partial x_2} & \cdots & \frac{\partial h}{\partial x_n} \\ \frac{\partial h}{\partial x_1} & \mathcal{L}_{x_1 x_1} & \mathcal{L}_{x_1 x_2} & \cdots & \mathcal{L}_{x_1 x_n} \\ \vdots & & & \ddots & \vdots \\ \frac{\partial h}{\partial x_n} & \mathcal{L}_{x_n x_1} & \cdots & & \mathcal{L}_{x_n x_n} \end{pmatrix}$$

where $\mathcal{L}_{x_i x_j} = \partial^2 \mathcal{L} / \partial x_i \partial x_j$ evaluated at $(x^*, \lambda^*)$.

**Notation for the test.** Let $\bar{D}_k$ denote the determinant of the leading $(k+1) \times (k+1)$ principal submatrix of $\bar{H}$ — i.e., the top-left $(k+1) \times (k+1)$ block (which always includes the border row and column). So:
- $\bar{D}_1 = \det\!\begin{pmatrix} 0 & h_{x_1} \\ h_{x_1} & \mathcal{L}_{x_1 x_1} \end{pmatrix} = -h_{x_1}^2$
- $\bar{D}_2 = \det$ of the $3 \times 3$ leading block
- $\vdots$
- $\bar{D}_n = \det(\bar{H})$ (full determinant)

The relevant minors are $\bar{D}_2, \bar{D}_3, \ldots, \bar{D}_n$ — a total of $n - 1$ of them. (We skip $\bar{D}_1$ because it's always non-positive and carries no information about the constrained curvature.)

**The bordered Hessian test (one equality constraint):**

| Condition | Sign pattern of $\bar{D}_2, \bar{D}_3, \ldots, \bar{D}_n$ |
|---|---|
| Sufficient for **local minimum** | All have sign $(-1)^1 = -1$, i.e., all **negative** |
| Sufficient for **local maximum** | Alternating signs $+, -, +, -,\ldots$ (i.e., $\bar{D}_k$ has sign $(-1)^{k}$) |

**For $m$ equality constraints** (bordered Hessian is $(n+m) \times (n+m)$, with $m$ border rows/columns): the relevant minors are of orders $2m+1$ through $n+m$, i.e., there are $n - m$ of them. Labeling them $\bar{\Delta}_1, \ldots, \bar{\Delta}_{n-m}$ in increasing order:

| Condition | Sign of $\bar{\Delta}_k$ |
|---|---|
| Sufficient for **local minimum** | All have sign $(-1)^m$, i.e., $\bar{\Delta}_k \cdot (-1)^m > 0$ for all $k$ |
| Sufficient for **local maximum** | $\bar{\Delta}_k$ has sign $(-1)^{m+k}$ |

**Why does this work? (Intuition, not full proof.)** The full proof uses the theory of constrained quadratic forms and the Schur complement identity — it is out of scope here, but is given in Nocedal & Wright §12.5. The intuition is: the bordered Hessian determinant encodes the signature (number of positive vs. negative eigenvalues) of $\nabla^2_x \mathcal{L}$ restricted to the tangent space $T_{x^*}$. The sign pattern $(-1)^m$ for local minima mirrors the standard Sylvester criterion for positive definiteness, shifted by the $m$ "negative directions" forced by the border rows.

### 3.5 Worked Example Continued

Returning to Section 2.6: $f(x) = \frac{1}{2} x^T Q x$ on $h(x) = a^T x - b = 0$.

$\nabla h = a$, $\nabla^2 h = 0$, so $\nabla^2_x \mathcal{L} = Q - \lambda^* \cdot 0 = Q$.

The bordered Hessian is:

$$\bar{H} = \begin{pmatrix} 0 & a^T \\ a & Q \end{pmatrix}$$

**Does the bordered Hessian test confirm a minimum?** Since $Q$ is SPD and the constraint is linear, we can directly check: for all $d \in T_{x^*} = \{d : a^T d = 0\}$:

$$d^T \nabla^2_x \mathcal{L} \, d = d^T Q d > 0$$

since $Q$ is positive definite on all of $\mathbb{R}^n$ (not just on $T_{x^*}$). So the second-order sufficient condition holds. $x^*$ is a local (in fact, global) minimum. ✓

---

## 4. Inequality Constraints: The KKT Conditions

### 4.1 The New Difficulty

The general constrained problem now includes inequality constraints:

$$\min_{x \in \mathbb{R}^n} f(x) \quad \text{s.t.} \quad g_j(x) \leq 0, \; j=1,\ldots,m \quad \text{and} \quad h_i(x) = 0, \; i=1,\ldots,p$$

**Why is this harder?** With equality constraints, every constraint is always "active" — the feasible set lives exactly on the surface. With inequality constraints, a constraint $g_j(x) \leq 0$ can be in one of two states at the optimum:

- **Active (tight):** $g_j(x^*) = 0$ — the optimum sits on the boundary of constraint $j$.
- **Inactive (slack):** $g_j(x^*) < 0$ — the optimum is in the strict interior; constraint $j$ is not binding.

We don't know in advance which constraints are active at the optimal point. If we knew, we could treat active constraints as equalities and apply Lagrange multipliers. Without that knowledge, we'd need to check all $2^m$ subsets of potentially active constraints — exponentially expensive.

The Karush-Kuhn-Tucker (KKT) conditions give a single set of necessary conditions that handle both active and inactive constraints simultaneously, via a clever mechanism called *complementary slackness*.

### 4.2 The KKT Lagrangian

**The extended Lagrangian** (Boyd & Vandenberghe convention):

$$\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_{i=1}^p \lambda_i h_i(x) + \sum_{j=1}^m \mu_j g_j(x)$$

where:
- $\lambda_i \in \mathbb{R}$: multipliers for equality constraints (free sign, as before)
- $\mu_j \geq 0$: multipliers for inequality constraints (must be non-negative — explained below)

**Sign convention note.** We use the $+$ convention for all terms. Since $\lambda_i$ is free, the equality constraint terms are unchanged in spirit. The sign of $\mu_j$ is meaningful and must be $\geq 0$; this is not a convention choice.

### 4.3 The Four KKT Conditions

A point $x^*$ satisfies the **KKT conditions** if there exist multipliers $\lambda^* \in \mathbb{R}^p$ and $\mu^* \in \mathbb{R}^m$ such that all of the following hold:

**Condition 1 — Stationarity:**

$$\nabla_x \mathcal{L}(x^*, \lambda^*, \mu^*) = \nabla f(x^*) + \sum_{i=1}^p \lambda_i^* \nabla h_i(x^*) + \sum_{j=1}^m \mu_j^* \nabla g_j(x^*) = 0$$

**Condition 2 — Primal feasibility:**

$$h_i(x^*) = 0 \; \forall i, \qquad g_j(x^*) \leq 0 \; \forall j$$

**Condition 3 — Dual feasibility:**

$$\mu_j^* \geq 0 \; \forall j$$

**Condition 4 — Complementary slackness:**

$$\mu_j^* \, g_j(x^*) = 0 \; \forall j$$

### 4.4 Unpacking Each Condition

**Stationarity.** This is the direct generalization of the Lagrange condition. The condition $\nabla f(x^*) + \sum \lambda_i^* \nabla h_i + \sum \mu_j^* \nabla g_j = 0$ rearranges to:

$$-\nabla f(x^*) = \sum_{i=1}^p \lambda_i^* \nabla h_i(x^*) + \sum_{j=1}^m \mu_j^* \nabla g_j(x^*)$$

In other words, $-\nabla f(x^*)$ (the direction of steepest *descent* of $f$, the direction we would move to decrease $f$) lies in the *cone* spanned by the active constraint gradients. Since $\mu_j^* \geq 0$, the inequality constraint contributions form a non-negative combination; the $\lambda_i^*$ are free-signed, so the equality constraint contributions form an unconstrained linear combination. Inactive inequality constraints contribute nothing ($\mu_j^* = 0$ by complementary slackness). The intuition: at a minimum, every direction that would decrease $f$ must be blocked by at least one constraint, and the constraint normals together "trap" the descent direction.

**Primal feasibility.** $x^*$ must actually lie in the feasible set. Stated for completeness.

**Dual feasibility ($\mu_j^* \geq 0$).** Here's the careful argument for why inequality multipliers must be non-negative.

Consider a single active inequality $g(x) \leq 0$ with $g(x^*) = 0$. The gradient $\nabla g(x^*)$ points in the direction of *increasing* $g$ — that is, it points *outward* from the feasible set (toward $g > 0$, the infeasible region). Think of $-\nabla g(x^*)$ as pointing inward into the feasible set.

For $x^*$ to be a minimum, there must be no feasible descent direction — no direction $d$ with $\nabla f(x^*)^T d < 0$ and $\nabla g(x^*)^T d \leq 0$ (the latter being the first-order condition for staying feasible near $x^*$).

From the stationarity condition (one constraint): $\nabla f(x^*) = -\mu \nabla g(x^*)$. 

If $\mu < 0$, then $\nabla f(x^*) = -\mu \nabla g(x^*) = |\mu| \nabla g(x^*)$ points *outward*. Then the direction $d = -\nabla g(x^*)$ (pointing inward, into the feasible set) satisfies:
$$\nabla f(x^*)^T d = |\mu| \nabla g(x^*)^T (-\nabla g(x^*)) = -|\mu| \|\nabla g\|^2 < 0$$
This is a feasible descent direction — contradicting $x^*$ being a minimum. So we need $\mu \geq 0$.

Geometrically: $-\nabla f = \mu \nabla g$ with $\mu \geq 0$ means the descent direction $-\nabla f$ is a non-negative multiple of the outward normal $\nabla g$ — the "push to decrease $f$" is directed into the infeasible region, blocked by the constraint.

**Complementary slackness ($\mu_j^* g_j(x^*) = 0$).** Since $g_j(x^*) \leq 0$ (primal feasibility) and $\mu_j^* \geq 0$ (dual feasibility), their product satisfies $\mu_j^* g_j(x^*) \leq 0$. The condition $\mu_j^* g_j(x^*) = 0$ pins this product to zero, forcing exactly one of the two factors to vanish. This means: for each $j$, either:

- $\mu_j^* = 0$: constraint $j$ is "switched off" — it plays no role in the stationarity condition regardless of whether it's active or not.
- $g_j(x^*) = 0$: constraint $j$ is active (tight) at the optimum.

**Why must this hold?** 

Case $g_j(x^*) < 0$ (inactive): In a neighborhood of $x^*$, the constraint is not binding — we can move in any direction and remain feasible with respect to this constraint. It imposes no local restriction, so it should not appear in the optimality condition. If $\mu_j^* > 0$, it would contribute $\mu_j^* \nabla g_j(x^*)$ to the stationarity condition — a nonzero influence from a non-binding constraint. This would give the wrong condition. So $\mu_j^* = 0$.

Case $g_j(x^*) = 0$ (active): The constraint is binding; $\mu_j^*$ may be nonzero and may need to be to make the stationarity condition hold.

The word *complementary* comes from the idea that $\mu_j^*$ and $g_j(x^*)$ are "complementary variables" — at most one of them can be nonzero.

**Putting it together.** Complementary slackness automates the identification of active constraints. Instead of guessing which constraints are active (and checking all $2^m$ subsets), the KKT conditions produce a system of equations that, when solved, simultaneously tells us both the optimal point *and* which constraints are active (those with $\mu_j^* > 0$).

### 4.5 Constraint Qualifications

**Why a qualification is needed.** The KKT conditions are *necessary* for optimality — but only under a regularity condition called a *constraint qualification* (CQ). A CQ rules out degenerate constraint geometries where the KKT argument breaks down. Without one, an optimal point might not satisfy the KKT conditions.

**A canonical failure example.** Consider: minimize $f(x_1, x_2) = x_1$ subject to:
$$g_1(x_1, x_2) = x_2 - x_1^3 \leq 0, \quad g_2(x_1, x_2) = -x_2 \leq 0$$

The feasible set is $\{(x_1, x_2) : 0 \leq x_2 \leq x_1^3\}$. For $x_1 < 0$, $x_1^3 < 0$, so the constraints $x_2 \leq x_1^3 < 0$ and $x_2 \geq 0$ are contradictory — there are no feasible points with $x_1 < 0$. The feasible set therefore has $x_1 \geq 0$, and the minimum of $f = x_1$ over this set is achieved at $x_1 = 0$, forced to $x_2 = 0$ by the constraints. So $x^* = (0,0)$ is the global minimum.

*(Note: if we had used $g_1 = x_2 - x_1^2$ instead, the feasible set would include points $(-\varepsilon, 0)$ for any $\varepsilon > 0$ — those are feasible with $f = -\varepsilon < 0$, so $(0,0)$ would not even be a local minimum. The cubic $x_1^3$ is essential here.)*

Both constraints are active at $x^*$: $g_1(0,0) = 0 - 0 = 0$, $g_2(0,0) = 0$. Their gradients are:

$$\nabla g_1 = (-3x_1^2, 1)\big|_{(0,0)} = (0, 1), \qquad \nabla g_2 = (0,-1)$$

The KKT stationarity condition requires $\nabla f + \mu_1 \nabla g_1 + \mu_2 \nabla g_2 = 0$:

$$(1, 0) + \mu_1 (0, 1) + \mu_2 (0,-1) = (0,0)$$
$$\implies 1 = 0 \quad \text{(from the first component)}$$

This is a contradiction. No KKT multipliers exist, even though $x^* = (0,0)$ is optimal. **The KKT conditions fail.**

What went wrong? $\nabla g_1 = (0,1)$ and $\nabla g_2 = (0,-1)$ are linearly dependent (one is $-1$ times the other). The two constraint normals cancel each other out, so no combination of them can produce $\nabla f = (1,0)$. This is a failure of the constraint geometry, not of the point being optimal.

**LICQ — Linear Independence Constraint Qualification.** Define the *active set* at $x^*$:

$$\mathcal{A}(x^*) = \{j : g_j(x^*) = 0\}$$

(the indices of constraints that are tight at $x^*$).

**LICQ holds at $x^*$** if the set of vectors:

$$\{\nabla g_j(x^*) : j \in \mathcal{A}(x^*)\} \cup \{\nabla h_i(x^*) : i = 1, \ldots, p\}$$

is *linearly independent* (no vector in this set is a linear combination of the others).

In the failure example: $\nabla g_1 = (0,1) = -\nabla g_2$, so linear independence fails.

**Consequence:** If LICQ holds at a local minimum $x^*$, then $x^*$ satisfies the KKT conditions and, moreover, the KKT multipliers $(\lambda^*, \mu^*)$ are unique.

**Slater's Condition** (specific to convex problems). Assume $f, g_1, \ldots, g_m$ are convex and $h_1, \ldots, h_p$ are affine. Slater's condition holds if there exists a **strictly feasible** point:

$$\exists \hat{x} \text{ s.t. } g_j(\hat{x}) < 0 \text{ for all } j = 1,\ldots,m \quad \text{and} \quad h_i(\hat{x}) = 0 \text{ for all } i = 1,\ldots,p$$

*Strictly feasible* means all inequality constraints hold with strict inequality (the point is in the interior of the feasible region, not on any inequality boundary).

If Slater's condition holds:
1. Any local (hence global) minimum $x^*$ satisfies the KKT conditions.
2. *Strong duality* holds: the primal and dual optimal values are equal (Section 5.3).

**Why is Slater's condition so useful in ML?** Most ML problems are convex with constraints like $\|w\|^2 \leq t$ (ridge) or probability simplex constraints. Strict feasibility is almost always obvious (e.g., $w = 0$ satisfies $\|w\|^2 < t$ for any $t > 0$), so Slater's gives KKT for free. In contrast, LICQ requires checking linear independence at the solution — which requires knowing the solution first.

**Hierarchy.** LICQ and Slater's are two different CQs; neither implies the other in general. For convex problems with strictly feasible points, Slater's is the standard tool. For general non-convex problems, LICQ is more commonly invoked.

### 4.6 Necessity vs. Sufficiency

**Necessity (general, non-convex case).** If $x^*$ is a local minimum and a CQ holds at $x^*$, then $x^*$ is a KKT point (i.e., KKT multipliers exist satisfying all four conditions).

Proof sketch: the argument extends the geometric intuition from equality constraints. With a CQ, the constraint gradients span the normal space correctly, and the optimality of $x^*$ forces $\nabla f(x^*)$ to lie in the cone generated by the active constraint gradients (Farkas' lemma), which gives stationarity. Complementary slackness and dual feasibility then follow from the specific form of the inequality constraints. The full proof is in Nocedal & Wright §12.3.

**Sufficiency (convex case).** If $f$ and all $g_j$ are convex, all $h_i$ are affine, and $x^*$ satisfies the KKT conditions, then $x^*$ is a global minimum.

*Why:* In a convex problem, the Lagrangian $\mathcal{L}(x, \lambda^*, \mu^*)$ is a convex function of $x$ — $f$ is convex by assumption, $\lambda_i^* h_i(x)$ is affine in $x$ (hence convex regardless of the sign of $\lambda_i^*$), and $\mu_j^* g_j(x)$ is a non-negative ($\mu_j^* \geq 0$) multiple of a convex function, hence convex. The sum of convex functions is convex. Stationarity says $\nabla_x \mathcal{L}(x^*, \lambda^*, \mu^*) = 0$, which for a convex function is the global minimum condition — so $x^*$ is a global minimizer of $\mathcal{L}(x, \lambda^*, \mu^*)$ over all $x \in \mathbb{R}^n$. Now write the chain of inequalities for any primal feasible $\tilde{x}$:

$$f(x^*) \underbrace{=}_{\text{comp. slack.}} \mathcal{L}(x^*, \lambda^*, \mu^*) \underbrace{\leq}_{\substack{x^* \text{ minimizes}\\\mathcal{L} \text{ over } \mathbb{R}^n}} \mathcal{L}(\tilde{x}, \lambda^*, \mu^*) = f(\tilde{x}) + \underbrace{\sum_i \lambda_i^* h_i(\tilde{x})}_{=\,0} + \underbrace{\sum_j \mu_j^* g_j(\tilde{x})}_{\leq\,0} \underbrace{\leq}_{\phantom{x}} f(\tilde{x})$$

The first equality uses complementary slackness (as in Section 5.2: $\sum_j \mu_j^* g_j(x^*) = 0$ and $h_i(x^*) = 0$). The last two inequalities use primal feasibility of $\tilde{x}$. Since this holds for all primal feasible $\tilde{x}$, $x^*$ is the global minimum.

**In non-convex problems,** KKT conditions are necessary but not sufficient. KKT points can be local minima, local maxima, or saddle points. Additional analysis (e.g., second-order conditions, or verifying global structure) is needed.

### 4.7 Worked Example: Quadratic with Inequality Constraint

**Problem.** Minimize $f(x,y) = (x-3)^2 + (y-2)^2$ subject to $x + y \leq 4$, $x \geq 0$, $y \geq 0$.

*Interpretation:* Find the nearest point in the triangle $\{x+y \leq 4,\; x \geq 0,\; y \geq 0\}$ to the target $(3,2)$.

Rewrite constraints in standard form $g_j \leq 0$:
$$g_1(x,y) = x + y - 4 \leq 0, \quad g_2(x,y) = -x \leq 0, \quad g_3(x,y) = -y \leq 0$$

**Lagrangian:**

$$\mathcal{L} = (x-3)^2 + (y-2)^2 + \mu_1(x+y-4) + \mu_2(-x) + \mu_3(-y)$$

**KKT stationarity:**

$$\frac{\partial \mathcal{L}}{\partial x} = 2(x-3) + \mu_1 - \mu_2 = 0$$
$$\frac{\partial \mathcal{L}}{\partial y} = 2(y-2) + \mu_1 - \mu_3 = 0$$

**Primal feasibility:** $x + y \leq 4$, $x \geq 0$, $y \geq 0$.

**Dual feasibility:** $\mu_1, \mu_2, \mu_3 \geq 0$.

**Complementary slackness** ($\mu_j^* g_j(x^*) = 0$ for each $j$):

$$\mu_1 \cdot g_1(x,y) = \mu_1(x+y-4) = 0$$
$$\mu_2 \cdot g_2(x,y) = \mu_2(-x) = 0 \quad \Longleftrightarrow \quad \mu_2 \cdot x = 0 \; \text{(since } -x = 0 \Leftrightarrow x = 0\text{)}$$
$$\mu_3 \cdot g_3(x,y) = \mu_3(-y) = 0 \quad \Longleftrightarrow \quad \mu_3 \cdot y = 0$$

**Solving.** First check whether the unconstrained minimum $(3,2)$ is feasible: $3+2 = 5 > 4$. Not feasible — constraint $g_1$ must be active.

**Try: $g_1$ active, $g_2, g_3$ inactive** (i.e., $x+y=4$, $\mu_2 = \mu_3 = 0$, and we'll verify $x, y > 0$ afterward).

Stationarity with $\mu_2 = \mu_3 = 0$:
$$2(x-3) + \mu_1 = 0 \implies x = 3 - \frac{\mu_1}{2}$$
$$2(y-2) + \mu_1 = 0 \implies y = 2 - \frac{\mu_1}{2}$$

Active constraint: $x + y = 4$:
$$\left(3 - \frac{\mu_1}{2}\right) + \left(2 - \frac{\mu_1}{2}\right) = 4 \implies 5 - \mu_1 = 4 \implies \mu_1^* = 1$$

So $x^* = 3 - \frac{1}{2} = \frac{5}{2}$, $y^* = 2 - \frac{1}{2} = \frac{3}{2}$.

**Verification:** $x^* = 5/2 > 0$ ✓, $y^* = 3/2 > 0$ ✓, $\mu_1^* = 1 \geq 0$ ✓. All KKT conditions satisfied.

**Solution:** $x^* = \frac{5}{2}$, $y^* = \frac{3}{2}$, with $\mu_1^* = 1$, $\mu_2^* = \mu_3^* = 0$.

**Shadow price interpretation.** Consider the perturbed problem with constraint $x + y \leq 4 + b$ for small $b$. By the envelope theorem (the same argument as §2.5 but applied at the constraint boundary where the constraint is active), the optimal value $p^*(b)$ satisfies:

$$\frac{dp^*}{db} = -\mu_1^*$$

The minus sign arises because the KKT Lagrangian uses the plus convention — $\mathcal{L} = f + \mu g$ — so the multiplier $\mu^*$ corresponds to $-\lambda^*$ in the minus-convention used in §2.5. With $\mu_1^* = 1$, relaxing the constraint by $\varepsilon$ (i.e., $b = \varepsilon$) changes the optimal squared distance by $-1 \cdot \varepsilon = -\varepsilon$: the minimum distance *decreases* by $\varepsilon$. Intuitively, loosening the constraint lets the feasible set expand, allowing us to get closer to the target $(3,2)$.

---

## 5. Lagrangian Duality

The KKT conditions tell us *what* an optimum looks like — but they say nothing about *how to find it*, or how to certify that a candidate solution is globally optimal (not just a local KKT point). Lagrangian duality addresses both. The key idea: by choosing the multipliers $(\lambda, \mu)$ cleverly, we can construct a *lower bound* on the optimal value of any minimization problem using only the Lagrangian — no feasibility check required. Maximizing this lower bound over all valid multipliers gives the dual problem, and when the gap between the two closes to zero, the dual solution simultaneously certifies global optimality of the primal.

### 5.1 The Dual Function

The Lagrangian relaxes constraints into the objective. For fixed multipliers $(\lambda, \mu)$, minimizing $\mathcal{L}(x, \lambda, \mu)$ over $x$ (ignoring the original constraints entirely) gives the *Lagrangian dual function*:

$$q(\lambda, \mu) = \inf_{x \in \mathbb{R}^n} \mathcal{L}(x, \lambda, \mu) = \inf_{x} \left[ f(x) + \sum_i \lambda_i h_i(x) + \sum_j \mu_j g_j(x) \right]$$

where $\inf$ (*infimum*) means the greatest lower bound — the minimum value if attained, or the limit approached if not. For many problems (e.g., quadratics), the infimum is attained and equals a minimum.

**Key property: $q$ is always concave in $(\lambda, \mu)$**, even if $f$ and the constraints are non-convex. This is because $q$ is a pointwise infimum of affine functions in $(\lambda, \mu)$ — each fixed $x$ gives the affine function $(\lambda, \mu) \mapsto f(x) + \sum_i \lambda_i h_i(x) + \sum_j \mu_j g_j(x)$, and the infimum of affines is concave.

### 5.2 Weak Duality

**Claim:** For any primal feasible $x$ and any dual feasible $(\lambda, \mu)$ (i.e., $\mu \geq 0$):

$$q(\lambda, \mu) \leq f(x)$$

**Proof.** Let $p^* = f(x^*)$ be the primal optimal value. Take any feasible $x$ and any $\mu \geq 0$. Then:

$$q(\lambda, \mu) = \inf_{\tilde{x}} \mathcal{L}(\tilde{x}, \lambda, \mu) \leq \mathcal{L}(x, \lambda, \mu) = f(x) + \underbrace{\sum_i \lambda_i h_i(x)}_{= 0, \text{ since } h_i(x) = 0} + \underbrace{\sum_j \mu_j g_j(x)}_{\leq 0, \text{ since } \mu_j \geq 0,\; g_j(x) \leq 0}$$

$$\leq f(x)$$

The first inequality uses $\inf \leq$ value at any specific point. The constraint terms vanish or are non-positive for feasible $x$ and dual-feasible $\mu$. $\square$

**So what?** Weak duality means $q(\lambda, \mu)$ is a lower bound on the primal optimal value $p^*$ for every dual-feasible $(\lambda, \mu)$. The *best* lower bound achievable by the dual is:

$$d^* = \sup_{\lambda,\; \mu \geq 0} q(\lambda, \mu) \leq p^*$$

This maximization of the dual function is the **dual problem**. Since $q$ is concave, the dual is always a convex optimization problem — even when the primal is non-convex. The quantity $p^* - d^*$ is the **duality gap**.

### 5.3 Strong Duality and Slater's Condition

**Strong duality** holds when $d^* = p^*$ — the duality gap is zero. This is not guaranteed in general (weak duality only gives $d^* \leq p^*$), but for convex problems with Slater's condition it always holds.

**Theorem (Strong duality under Slater's condition).** If $f$ and $g_1, \ldots, g_m$ are convex, $h_1, \ldots, h_p$ are affine, and Slater's condition holds, then $d^* = p^*$, and the dual optimal value is attained by some $(\lambda^*, \mu^*)$.

The proof is non-trivial (it uses the separating hyperplane theorem for convex sets) and is out of scope here; see Boyd & Vandenberghe §5.3.2.

**Why does strong duality matter?** Two reasons:

1. **Computational:** The dual problem is always convex (even if the primal isn't, though with Slater's the primal is convex here). Sometimes the dual is easier to solve — it has fewer variables, simpler structure, or admits a closed form. Solving the dual and recovering the primal solution is a common strategy in SVMs and other ML methods.

2. **Optimality certificates:** At $d^* = p^*$, the dual optimal $(\lambda^*, \mu^*)$ provides a *certificate* that $x^*$ is globally optimal — without checking all feasible points. This is important in large-scale optimization.

### 5.4 KKT as Primal-Dual Optimality

When strong duality holds, the KKT conditions characterize the simultaneous optimality of the primal and dual problems:

**Claim:** If Slater's condition holds, then $x^*$ is primal optimal and $(\lambda^*, \mu^*)$ is dual optimal *if and only if* $(x^*, \lambda^*, \mu^*)$ satisfies the KKT conditions.

*Why:* (a) If KKT holds, we need to show $x^*$ minimizes $\mathcal{L}(x, \lambda^*, \mu^*)$ over $x$. Here we use the convexity assumed by Slater's condition: $\mathcal{L}(\cdot, \lambda^*, \mu^*)$ is convex in $x$ (same argument as Section 4.6 sufficiency). Therefore stationarity $\nabla_x \mathcal{L}(x^*, \lambda^*, \mu^*) = 0$ implies $x^*$ is a global minimizer of $\mathcal{L}$, so $q(\lambda^*, \mu^*) = \mathcal{L}(x^*, \lambda^*, \mu^*)$. Complementary slackness and primal feasibility then give $\mathcal{L}(x^*, \lambda^*, \mu^*) = f(x^*)$. So $q(\lambda^*, \mu^*) = f(x^*)$ — the duality gap is zero. (b) Conversely, if $d^* = p^*$ and both optima are attained, the conditions that make the chain of inequalities in the weak duality proof hold with equality force exactly the KKT conditions.

This is why KKT conditions are so central: **they are the joint optimality conditions for the primal-dual pair.** Solving the KKT system simultaneously finds the primal solution $x^*$ and the dual certificate $(\lambda^*, \mu^*)$.

---

## 6. ML Applications

### 6.1 Regularization as a Constraint

**Ridge regression, two ways.** Ridge regression is typically written in *penalized form*:

$$\min_{w \in \mathbb{R}^d} \|y - Xw\|^2 + \lambda \|w\|^2 \quad \quad (\text{Penalized Ridge})$$

There is an equivalent *constrained form*:

$$\min_{w \in \mathbb{R}^d} \|y - Xw\|^2 \quad \text{s.t.} \quad \|w\|^2 \leq t \quad \quad (\text{Constrained Ridge})$$

We will now show these are equivalent (in the sense that for each $\lambda \geq 0$ there exists $t \geq 0$ such that both have the same solution), and in doing so, reveal that the regularization parameter is literally a Lagrange multiplier.

**KKT analysis of Constrained Ridge.** Write the constraint as $g(w) = \|w\|^2 - t \leq 0$.

Lagrangian:

$$\mathcal{L}(w, \mu) = \|y - Xw\|^2 + \mu(\|w\|^2 - t)$$

Stationarity ($\nabla_w \mathcal{L} = 0$):

$$-2X^T(y - Xw) + 2\mu w = 0$$

$$X^T y - X^T X w = \mu w$$

$$(X^T X + \mu I) w = X^T y$$

$$w^* = (X^T X + \mu I)^{-1} X^T y$$

*Why is $X^TX + \mu I$ invertible?* For any $\mu > 0$: $X^TX$ is positive semidefinite (since $v^T X^T X v = \|Xv\|^2 \geq 0$), and $\mu I$ is positive definite. Their sum is positive definite, hence invertible. The regularization term $\mu I$ is precisely what rescues invertibility when $X$ does not have full column rank — this is one of ridge regression's key practical advantages over OLS.

This is exactly the ridge regression solution with $\mu$ in place of $\lambda$. **The regularization parameter is the Lagrange multiplier for the norm constraint.**

**KKT complementary slackness:** $\mu^*(\|w^*\|^2 - t) = 0$. Two regimes:

1. **Constraint inactive ($\|w_{\text{OLS}}\|^2 \leq t$):** The unconstrained OLS solution $w_{\text{OLS}} = (X^TX)^{-1}X^Ty$ satisfies the norm bound. Then $\mu^* = 0$ and $w^* = w_{\text{OLS}}$ — no regularization.

2. **Constraint active ($\|w^*\|^2 = t$):** The OLS solution violates the bound; the optimal $w^*$ lies on the boundary of the ball $\|w\|^2 = t$, with $\mu^* > 0$ — regularization is in effect.

**Slater's condition check.** Take $w = 0$: $g(0) = -t < 0$ for any $t > 0$. Strictly feasible. ✓ So KKT is necessary and sufficient, and strong duality holds.

**The conceptual payoff.** Regularization is not an ad hoc trick — it is the Lagrange multiplier for a constraint on model complexity. The regularization parameter $\lambda$ encodes the *shadow price* of the norm ball: it quantifies how much you must degrade the training fit to shrink the model by one unit. This reframing:

- Explains why $\lambda \to 0$ recovers OLS (constraint becomes non-binding).
- Explains why $\lambda \to \infty$ drives $w \to 0$ (constraint shrinks to a point).
- Generalizes: Lasso ($L^1$ penalty) corresponds to a constraint $\|w\|_1 \leq t$; elastic net corresponds to a combination; in each case, the penalty parameter is the shadow price of the respective constraint.

### 6.2 Maximum Entropy and the Exponential Family

**The motivation.** Suppose we want to model a probability distribution $p = (p_1, \ldots, p_n)$ over $n$ outcomes, but we only know certain statistics — for example, the expected values of $K$ features: $\sum_i p_i f_k(i) = c_k$ for $k = 1, \ldots, K$. We have a whole family of distributions consistent with these constraints. Which one should we use?

The **maximum entropy principle** (Jaynes, 1957) says: choose the distribution that has maximum *Shannon entropy* subject to the known constraints. The intuition is that this is the "least informative" distribution consistent with what we know — it spreads probability as uniformly as possible while respecting the constraints, avoiding the injection of assumptions we have no evidence for.

**Shannon entropy** of a discrete distribution $p$ over $\{1, \ldots, n\}$:

$$H(p) = -\sum_{i=1}^n p_i \log p_i$$

*Note:* The convention $0 \log 0 := 0$ handles zero probabilities continuously. $H(p) \geq 0$ always, and $H$ is maximized (at $H = \log n$) by the uniform distribution $p_i = 1/n$ — maximum uncertainty corresponds to maximum entropy.

**The problem.** We want to maximize $H(p)$ — equivalently, minimize $-H(p) = \sum_i p_i \log p_i$ (a convex function):

$$\min_{p \in \mathbb{R}^n} \sum_{i=1}^n p_i \log p_i \quad \text{s.t.} \quad \underbrace{\sum_{i=1}^n p_i = 1}_{h_0(p)=0}, \quad \underbrace{\sum_{i=1}^n p_i f_k(i) = c_k}_{h_k(p)=0} \text{ for } k=1,\ldots,K$$

(We also implicitly need $p_i \geq 0$, i.e., $g_i(p) = -p_i \leq 0$ for each $i$; we'll see these don't bind at the solution.)

**Lagrangian:**

$$\mathcal{L}(p, \lambda_0, \lambda) = \sum_{i=1}^n p_i \log p_i - \lambda_0\!\left(\sum_i p_i - 1\right) - \sum_{k=1}^K \lambda_k\!\left(\sum_i p_i f_k(i) - c_k\right)$$

where $\lambda_0$ is the multiplier for normalization and $\lambda_k$ are multipliers for the moment constraints.

**Stationarity ($\partial \mathcal{L}/\partial p_i = 0$):**

We need $\frac{\partial}{\partial p_i} \sum_j p_j \log p_j$. Using $\frac{d}{dp}[p \log p] = \log p + 1$:

$$\frac{\partial \mathcal{L}}{\partial p_i} = \log p_i + 1 - \lambda_0 - \sum_{k=1}^K \lambda_k f_k(i) = 0$$

Solve for $\log p_i$:

$$\log p_i = \lambda_0 - 1 + \sum_{k=1}^K \lambda_k f_k(i)$$

Exponentiate both sides (the exponential of a sum is the product of exponentials):

$$p_i^* = e^{\lambda_0 - 1} \cdot \exp\!\left(\sum_{k=1}^K \lambda_k f_k(i)\right) = \frac{1}{Z} \exp\!\left(\sum_{k=1}^K \lambda_k f_k(i)\right)$$

where the **partition function** $Z := e^{1 - \lambda_0}$ is a normalizing constant, uniquely determined by the normalization constraint $\sum_i p_i^* = 1$:

$$Z = \sum_{i=1}^n \exp\!\left(\sum_{k=1}^K \lambda_k f_k(i)\right)$$

**This is the Gibbs/Boltzmann distribution** — the canonical distribution of statistical mechanics, and the core of the exponential family in statistics.

**Reading the result.** Let's annotate the final distribution:

$$\underbrace{p_i^*}_{\text{optimal probability of outcome }i} = \underbrace{\frac{1}{Z}}_{\text{normalization}} \exp\!\underbrace{\left(\sum_{k=1}^K \lambda_k f_k(i)\right)}_{\text{weighted sum of features at outcome }i}$$

- The features $f_k(i)$ encode what we know about outcome $i$.
- The Lagrange multipliers $\lambda_k$ are the *natural parameters* — they calibrate how much each known constraint shapes the distribution. A large $|\lambda_k|$ means constraint $k$ pulls the distribution strongly; $\lambda_k = 0$ means constraint $k$ is irrelevant.
- The partition function $Z$ normalizes probabilities. It also plays a central role in computing the log-likelihood and its gradients.

**Checking the non-negativity constraints.** Since $p_i^* = \frac{1}{Z} e^{(\cdot)}$ with $Z > 0$, we have $p_i^* > 0$ always. The inequality constraints $-p_i \leq 0$ are inactive at the solution (as expected for smooth interior optima). ✓

**Special cases and connections:**

- **No moment constraints** (only $\sum p_i = 1$): Then $\lambda_k = 0$ for all $k$, and $p_i^* = 1/Z = 1/n$ — the uniform distribution. Maximum entropy with no information is uniform. ✓

- **One second-moment (variance) constraint** $\sum_i p_i x_i^2 = \sigma^2$ (continuous limit, zero-mean case): Gives the Gaussian $p(x) \propto e^{-\lambda x^2}$, where $\lambda = 1/(2\sigma^2)$. To see this: the MaxEnt stationarity condition with a single constraint $\mathbb{E}[X^2] = \sigma^2$ gives $\log p(x) + 1 - \lambda_0 - \lambda x^2 = 0$, i.e., $p(x) \propto e^{-\lambda x^2}$, which is the Gaussian $N(0, 1/(2\lambda))$. Matching $\mathbb{E}[X^2] = \sigma^2$ gives $\lambda = 1/(2\sigma^2)$. More generally, constraining both the mean $\mathbb{E}[X] = \mu$ and the variance $\mathrm{Var}(X) = \sigma^2$ gives $p(x) \propto e^{-\lambda_1 x - \lambda_2 x^2} = N(\mu, \sigma^2)$ for appropriate $\lambda_1, \lambda_2$. *(A mean constraint alone — fixing only $\mathbb{E}[X] = \mu$ — does not give the Gaussian; on the real line it gives $p \propto e^{\lambda x}$, which is not normalizable, and on a bounded support it gives an exponential distribution. The Gaussian requires constraining the second moment.)* This is why the Gaussian is the "maximum ignorance" distribution given only mean and variance information.

- **Logistic regression:** In multi-class classification with linear features, the softmax output $P(y=k|x) \propto \exp(\theta_k^T x)$ is exactly the MaxEnt distribution over classes, with $\theta_k^T x$ as the feature values. Logistic regression is MaxEnt.

**The conceptual payoff.** The exponential family (Gaussian, Bernoulli, Poisson, Gamma, etc.) can be derived axiomatically from the maximum entropy principle — it is the family of distributions that are *maximally non-committal* given constraints on certain statistics. The Lagrange multipliers are the natural parameters of the family. When you encounter exponential families in probabilistic ML (conjugate priors, generalized linear models, variational inference), the Lagrangian structure is always in the background.

---

## 7. Summary and Quick Reference

### Problem Taxonomy

| Problem | Method | Key equation |
|---|---|---|
| $\min f(x)$ s.t. $h(x) = 0$ | Equality Lagrangian | $\nabla f = \lambda \nabla h$, $h = 0$ |
| + second-order test | Bordered Hessian | $\bar{D}_k$ signs: all $(-1)^m$ for min |
| $\min f(x)$ s.t. $g(x) \leq 0$, $h(x)=0$ | KKT conditions | 4 conditions: stationarity, primal feas., dual feas., comp. slack. |
| Convex + Slater's | KKT = necessary and sufficient | $d^* = p^*$; KKT ↔ global optimality |

### The Four KKT Conditions (Quick Reference)

1. **Stationarity:** $\nabla f(x^*) + \sum \lambda_i^* \nabla h_i(x^*) + \sum \mu_j^* \nabla g_j(x^*) = 0$
2. **Primal feasibility:** $h_i(x^*) = 0$, $g_j(x^*) \leq 0$
3. **Dual feasibility:** $\mu_j^* \geq 0$
4. **Complementary slackness:** $\mu_j^* g_j(x^*) = 0$

### The Lagrange Multiplier as Shadow Price

For $\min f(x)$ s.t. $g(x) = c$: the optimal value $p^*(c)$ satisfies $dp^*/dc = \lambda^*$. Tightening a constraint by $\varepsilon$ changes the optimal value by $\approx \lambda^* \varepsilon$.

### ML Connections

| ML concept | Lagrange structure |
|---|---|
| Ridge/Lasso penalty | $\lambda$ = shadow price of norm constraint |
| Softmax / logistic regression | MaxEnt distribution (exponential family) |
| Exponential family natural parameters | Lagrange multipliers for moment constraints |
| SVM (hard-margin) | KKT conditions identify support vectors via comp. slackness |
| Variational inference (ELBO) | Lagrangian relaxation of probability constraints |

---

## 8. Sources and Further Reading

*Assigned readings:* None specified — add course-assigned readings here.

**Primary references:**

- **Boyd & Vandenberghe, *Convex Optimization* (2004), Chapters 4–5.** The standard ML/engineering reference. The treatment of KKT conditions, Slater's condition, and duality in Sections 4 and 5 closely follows B&V's framing. Available free at [https://web.stanford.edu/~boyd/cvxbook/](https://web.stanford.edu/~boyd/cvxbook/).

- **Nocedal & Wright, *Numerical Optimization*, 2nd ed. (2006), Chapters 12–13.** The rigorous treatment of constraint qualifications, second-order conditions, and bordered Hessian theory follows this source. The LICQ failure example in Section 4.5 is in the spirit of their §12.2.

- **Bertsekas, *Nonlinear Programming*, 3rd ed. (2016).** Comprehensive proofs of KKT necessity theorems (Propositions 3.3.1–3.3.4). The shadow price result (Section 2.5) is proved rigorously in §3.2.

**For specific topics:**

- **Jaynes (1957), "Information Theory and Statistical Mechanics," *Physical Review* 106(4):620–630.** The original paper deriving the maximum entropy principle and the Boltzmann distribution as a Lagrange multiplier result (Section 6.2).

- **Hastie, Tibshirani & Friedman, *Elements of Statistical Learning*, 2nd ed. (2009), §3.4.** The ridge-as-constraint equivalence and the path of solutions as a function of $\lambda$ (Section 6.1).

- The shadow price result (Section 2.5) and the equivalence between regularization and constraints (Section 6.1) are standard results not cleanly attributable to a single source; they appear in substantially all graduate-level optimization texts.

- The bordered Hessian sign conditions (Section 3.4) follow Chiang & Wainwright, *Fundamental Methods of Mathematical Economics*, 4th ed., §11.4 — which gives the clearest presentation of the sign rules with concrete examples.
