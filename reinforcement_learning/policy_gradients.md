# Policy Gradient Methods: From REINFORCE to PPO and SAC

---

## Table of Contents

- [Policy Gradient Methods: From REINFORCE to PPO and SAC](#policy-gradient-methods-from-reinforce-to-ppo-and-sac)
  - [Table of Contents](#table-of-contents)
  - [1. Why Parameterize the Policy Directly?](#1-why-parameterize-the-policy-directly)
    - [1.1 The Limits of Value-Based Methods](#11-the-limits-of-value-based-methods)
    - [1.2 The Policy Gradient Idea](#12-the-policy-gradient-idea)
    - [1.3 Value-Based vs. Policy-Based vs. Actor-Critic](#13-value-based-vs-policy-based-vs-actor-critic)
  - [2. The Policy Gradient Theorem](#2-the-policy-gradient-theorem)
    - [2.1 Setup: Parameterized Policies and the Objective](#21-setup-parameterized-policies-and-the-objective)
    - [2.2 Deriving the Policy Gradient](#22-deriving-the-policy-gradient)
    - [2.3 Why This Is Remarkable](#23-why-this-is-remarkable)
  - [3. REINFORCE: The Simplest Policy Gradient](#3-reinforce-the-simplest-policy-gradient)
    - [3.1 The Algorithm](#31-the-algorithm)
    - [3.2 A Worked Example](#32-a-worked-example)
    - [3.3 The Variance Problem](#33-the-variance-problem)
  - [4. Variance Reduction: Baselines and Advantage Functions](#4-variance-reduction-baselines-and-advantage-functions)
    - [4.1 The Baseline Trick](#41-the-baseline-trick)
    - [4.2 The Advantage Function](#42-the-advantage-function)
    - [4.3 Generalized Advantage Estimation (GAE)](#43-generalized-advantage-estimation-gae)
  - [5. Actor-Critic Methods](#5-actor-critic-methods)
    - [5.1 The Core Idea: Don't Wait for the Episode to End](#51-the-core-idea-dont-wait-for-the-episode-to-end)
    - [5.2 A2C: Advantage Actor-Critic](#52-a2c-advantage-actor-critic)
    - [5.3 A3C: Asynchronous Advantage Actor-Critic](#53-a3c-asynchronous-advantage-actor-critic)
    - [5.4 Connection to the Adaptive Heuristic Critic](#54-connection-to-the-adaptive-heuristic-critic)
  - [6. Trust Regions: Why Step Size Matters in Policy Space](#6-trust-regions-why-step-size-matters-in-policy-space)
    - [6.1 The Policy Collapse Problem](#61-the-policy-collapse-problem)
    - [6.2 Measuring Policy Change: KL Divergence](#62-measuring-policy-change-kl-divergence)
    - [6.3 TRPO: Trust Region Policy Optimization](#63-trpo-trust-region-policy-optimization)
  - [7. PPO: Proximal Policy Optimization](#7-ppo-proximal-policy-optimization)
    - [7.1 The Clipped Surrogate Objective](#71-the-clipped-surrogate-objective)
    - [7.2 The Full Algorithm](#72-the-full-algorithm)
    - [7.3 Why PPO Became the Default](#73-why-ppo-became-the-default)
  - [8. Deterministic Policy Gradients and DDPG](#8-deterministic-policy-gradients-and-ddpg)
    - [8.1 The DPG Theorem](#81-the-dpg-theorem)
    - [8.2 DDPG: Deep Deterministic Policy Gradient](#82-ddpg-deep-deterministic-policy-gradient)
  - [9. SAC: Soft Actor-Critic](#9-sac-soft-actor-critic)
    - [9.1 Maximum Entropy Reinforcement Learning](#91-maximum-entropy-reinforcement-learning)
    - [9.2 The SAC Architecture](#92-the-sac-architecture)
    - [9.3 Automatic Temperature Tuning](#93-automatic-temperature-tuning)
    - [9.4 Why SAC Matters](#94-why-sac-matters)
  - [10. Connections to Modern Applications](#10-connections-to-modern-applications)
    - [10.1 RLHF: Reinforcement Learning from Human Feedback](#101-rlhf-reinforcement-learning-from-human-feedback)
    - [10.2 Other Frontiers](#102-other-frontiers)
  - [11. The Full RL Landscape](#11-the-full-rl-landscape)
    - [11.1 How the Three Documents Connect](#111-how-the-three-documents-connect)
    - [11.2 Decision Flowchart: Which Algorithm Should I Use?](#112-decision-flowchart-which-algorithm-should-i-use)
  - [Sources and Further Reading](#sources-and-further-reading)
    - [Foundational policy gradient theory](#foundational-policy-gradient-theory)
    - [Actor-critic and advantage estimation](#actor-critic-and-advantage-estimation)
    - [Trust regions and proximal methods](#trust-regions-and-proximal-methods)
    - [Deterministic and continuous-action methods](#deterministic-and-continuous-action-methods)
    - [Maximum entropy RL](#maximum-entropy-rl)
    - [Applications and modern extensions](#applications-and-modern-extensions)
    - [Textbooks and surveys](#textbooks-and-surveys)

---

## 1. Why Parameterize the Policy Directly?

The companion documents — *RL Foundations* and *RL in Practice* — developed a complete toolkit for **value-based** reinforcement learning: learn $Q^*(s, a)$ (or $V^*(s)$), then derive the optimal policy via $\pi^*(s) = \arg\max_a Q^*(s, a)$. This approach, scaled through DQN and Rainbow, achieves superhuman performance on Atari games and works beautifully for problems with discrete action spaces.

But it hits a wall in several important settings. Understanding *where* value-based methods break down motivates the entire family of algorithms in this document.

### 1.1 The Limits of Value-Based Methods

**Continuous action spaces.** Consider a robotic arm that must apply a torque $a \in [-2.0, +2.0]$ Nm to each of its six joints. The action is a 6-dimensional continuous vector — there is no finite action set to take $\arg\max$ over. Value-based methods require evaluating $Q(s, a)$ for every candidate action and picking the best one. With discrete actions (left, right, up, down), this is a cheap comparison. With continuous actions, it becomes an optimization problem *inside* every time step: $\max_{a \in \mathbb{R}^6} Q(s, a)$. For a nonlinear neural network $Q$, this inner optimization has no closed-form solution and would require iterative optimization at every step — far too expensive.

**Stochastic policies.** Value-based methods produce deterministic policies: $\pi(s) = \arg\max_a Q(s, a)$ always maps to a single action. But some problems *require* stochastic policies. In rock-paper-scissors, any deterministic policy is exploitable — the only Nash equilibrium is the uniform random policy $\pi(\text{rock}) = \pi(\text{paper}) = \pi(\text{scissors}) = 1/3$. More generally, in partially observable environments or multi-agent settings, stochastic policies can be strictly better than any deterministic policy (Jaakkola et al., 1995, as discussed in *RL in Practice*, Section 5.2).

**Policy structure.** Sometimes the optimal policy is far simpler than the optimal value function. Imagine a high-dimensional observation space (raw pixels) where the best action in every state is simply "move toward the brightest region." The Q-function must assign a precise scalar value to every pixel-action combination — a massively complex surface. The policy, by contrast, can be a simple function from pixel features to a direction. Learning the policy directly is more natural when the policy has a simpler structure than the value function.

### 1.2 The Policy Gradient Idea

The key insight is deceptively simple: instead of learning $Q^*$ and deriving a policy from it, **parameterize the policy directly** and optimize it by gradient ascent on expected return.

Define a parameterized policy $\pi_\theta(a \mid s)$ — a function that maps states to probability distributions over actions, controlled by parameters $\theta$ (e.g., the weights of a neural network). The objective is the expected return when the agent follows $\pi_\theta$:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is a trajectory sampled by following $\pi_\theta$.

If we can compute $\nabla_\theta J(\theta)$ — the gradient of the expected return with respect to the policy parameters — then we can improve the policy by gradient ascent: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$. Each step nudges $\theta$ in the direction that most increases the expected return.

This sidesteps all three limitations above. There is no $\arg\max$ over actions, so continuous actions are natural — the network just outputs the parameters of a continuous distribution (e.g., mean and standard deviation of a Gaussian). The policy is inherently stochastic (it outputs a probability distribution), so stochastic optimal policies are representable. And we learn the policy directly, so if the policy has simple structure, the network can exploit it.

### 1.3 Value-Based vs. Policy-Based vs. Actor-Critic

Before diving into the algorithms, it helps to see the three families side by side:

| | **Value-Based** | **Policy-Based** | **Actor-Critic** |
|---|---|---|---|
| **What is learned** | $Q^*(s,a)$ or $V^*(s)$ | $\pi_\theta(a \mid s)$ directly | Both: $\pi_\theta$ (actor) + $V_\phi$ (critic) |
| **Policy extraction** | $\arg\max_a Q(s,a)$ | Sample from $\pi_\theta(\cdot \mid s)$ | Sample from $\pi_\theta(\cdot \mid s)$ |
| **Action spaces** | Discrete (finite $\arg\max$) | Discrete or continuous | Discrete or continuous |
| **Stochastic policies** | No (deterministic $\arg\max$) | Yes (native) | Yes (native) |
| **Variance** | Low (bootstraps off value estimates) | High (uses Monte Carlo returns) | Medium (critic reduces variance) |
| **Bias** | Low (converges to $Q^*$) | None (unbiased gradient) | Some (critic introduces bias) |
| **Key algorithms** | Q-learning, DQN, Rainbow | REINFORCE | A2C, PPO, SAC |

Actor-critic methods combine the best of both worlds: the actor provides a parameterized policy (handling continuous actions and stochastic policies), while the critic provides a value estimate that dramatically reduces the variance of the policy gradient. Nearly all modern policy gradient algorithms — A2C, PPO, SAC — are actor-critic methods. Pure policy-based methods like REINFORCE are primarily of pedagogical importance: they reveal the core ideas clearly, but their high variance makes them impractical for all but the simplest problems.

---

## 2. The Policy Gradient Theorem

The mathematical foundation of every algorithm in this document is the **policy gradient theorem** (Sutton et al., 1999). It tells us how to compute $\nabla_\theta J(\theta)$ — the direction in parameter space that most increases the expected return — using only quantities we can estimate from sample trajectories.

### 2.1 Setup: Parameterized Policies and the Objective

A **parameterized policy** $\pi_\theta(a \mid s)$ is a function that, given a state $s$, outputs a probability distribution over actions $a$, controlled by parameters $\theta$. For discrete actions, a common choice is a **softmax policy**: a neural network takes $s$ as input, produces a score (logit) $h_\theta(s, a)$ for each action, and the policy is:

$$\pi_\theta(a \mid s) = \frac{\exp(h_\theta(s, a))}{\sum_{a'} \exp(h_\theta(s, a'))}$$

For continuous actions, a common choice is a **Gaussian policy**: the network outputs a mean $\mu_\theta(s)$ and (typically) a diagonal covariance — e.g. a standard deviation $\sigma_\theta(s)$ per action dimension — and the action is sampled from $\mathcal{N}(\mu_\theta(s),\, \operatorname{diag}(\sigma_\theta(s)^2))$ (a full $d \times d$ covariance is possible but adds many parameters and is less common).

The **objective** is the expected return under $\pi_\theta$. In the episodic setting (the most common for policy gradients), this is:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau)\right] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

where $\tau = (s_0, a_0, r_0, s_1, \ldots, s_T)$ is a full trajectory sampled by following $\pi_\theta$ in the environment. The expectation is over all sources of randomness: the initial state distribution, the stochastic transitions, and the stochastic action selection.

Our goal: compute $\nabla_\theta J(\theta)$ so we can perform gradient ascent.

### 2.2 Deriving the Policy Gradient

The probability of a trajectory $\tau$ under policy $\pi_\theta$ is:

$$P(\tau \mid \theta) = p(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) \, T(s_{t+1} \mid s_t, a_t)$$

where $p(s_0)$ is the initial state distribution and $T(s_{t+1} \mid s_t, a_t)$ is the environment's transition function. The objective can then be written as:

$$J(\theta) = \sum_\tau P(\tau \mid \theta) \, R(\tau)$$

where the sum is over all possible trajectories (in practice, we'll estimate this sum via sampling). Now take the gradient with respect to $\theta$:

$$\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau \mid \theta) \, R(\tau)$$

The return $R(\tau)$ does not depend on $\theta$ (it's just a sum of rewards along a fixed trajectory), so the gradient passes through. The problem is that $\nabla_\theta P(\tau \mid \theta)$ is hard to work with directly — it's the gradient of a product of many terms.

The **log-derivative trick** (also called the score function trick or REINFORCE trick) converts this into something tractable. The key identity is:

$$\nabla_\theta P(\tau \mid \theta) = P(\tau \mid \theta) \, \nabla_\theta \log P(\tau \mid \theta)$$

This follows from the chain rule: $\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)}$, so $\nabla_\theta f(\theta) = f(\theta) \, \nabla_\theta \log f(\theta)$.

Substituting:

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau \mid \theta) \, \nabla_\theta \log P(\tau \mid \theta) \, R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\nabla_\theta \log P(\tau \mid \theta) \, R(\tau)\right]$$

Now expand $\log P(\tau \mid \theta)$:

$$\log P(\tau \mid \theta) = \log p(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log T(s_{t+1} \mid s_t, a_t)$$

When we differentiate with respect to $\theta$, the initial state distribution $p(s_0)$ and the transition function $T$ vanish — they don't depend on $\theta$. Only the policy terms survive:

$$\nabla_\theta \log P(\tau \mid \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

This is the crucial step. The environment dynamics $T$ — which the agent doesn't know — have dropped out of the gradient entirely.

**Result: the policy gradient theorem (trajectory form)**

$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, R(\tau)\right]}$$

Reading this in plain English: "To improve the policy, collect trajectories, and for each action taken, nudge the parameters to make that action more likely if the trajectory was good (high $R(\tau)$) and less likely if the trajectory was bad (low $R(\tau)$)." The term $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ is the direction in parameter space that would increase the probability of action $a_t$ in state $s_t$. The return $R(\tau)$ weights this direction — large positive returns amplify the signal, while negative returns reverse it.

A more refined version uses the **reward-to-go** $G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$ instead of the full trajectory return. The intuition is that action $a_t$ can only affect rewards from time $t$ onward — it cannot influence rewards that already happened at times $0, \ldots, t-1$. Using the full return adds noise without information. Replacing $R(\tau)$ with $G_t$ yields:

**Result: the policy gradient theorem (reward-to-go form)**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t\right]$$

**Why the expectation is unchanged.** Under the MDP factorization, past rewards $r_0, \ldots, r_{t-1}$ do not depend on $a_t$ given $s_t$. For each $t$, the extra terms you would get if you multiplied $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ by those past rewards have **zero expectation**, so they drop out of $\nabla_\theta J$ while inflating variance. What remains correlated with the score in the right way is exactly the **reward-to-go** $G_t$. Equivalently, one derives the same **per-timestep** policy gradient by starting from $J(\theta) = \mathbb{E}[\sum_t \gamma^t r_t]$ and pushing the gradient through the policy factors that influence each $r_{t'}$. See Sutton & Barto (2018), §13.3.

This is the form used in practice — same expected gradient, lower variance.

### 2.3 Why This Is Remarkable

Three properties make the policy gradient theorem powerful:

1. **Model-free.** The environment dynamics $T$ do not appear anywhere in the gradient. The agent only needs to sample trajectories from the environment and evaluate returns — no model of $T$ or $R$ is required. This is in the same spirit as Q-learning, but applied to the policy directly.

2. **Works with any differentiable policy.** The gradient only requires $\nabla_\theta \log \pi_\theta(a \mid s)$, which exists for any policy parameterized by a differentiable function (neural networks, linear models, etc.). It does not require the environment to be differentiable.

3. **Handles continuous actions natively.** There is no $\arg\max$ or $\max$ anywhere. For a (typically diagonal) Gaussian policy, the log-probability and its gradient are closed-form expressions — so continuous actions are as easy as discrete ones.

The price is **variance**. Because we estimate the gradient from sampled trajectories, the estimate is noisy. A single trajectory might happen to get a high return due to lucky transitions rather than good actions, and the gradient update would incorrectly reinforce those actions. Reducing this variance is the central challenge of practical policy gradient methods, and it drives the progression from REINFORCE (Section 3) through actor-critic methods (Section 5) to PPO (Section 7).

---

## 3. REINFORCE: The Simplest Policy Gradient

REINFORCE (Williams, 1992) is the most direct application of the policy gradient theorem. It collects complete trajectories, computes returns, and performs gradient ascent. Despite its simplicity, it establishes the template that every subsequent algorithm refines.

### 3.1 The Algorithm

**Algorithm: REINFORCE**

> Initialize policy parameters $\theta$ randomly.
>
> Repeat:
>
> $\quad$ Sample a trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$ by following $\pi_\theta$.
>
> $\quad$ For each time step $t = 0, 1, \ldots, T-1$:
>
> $\qquad$ Compute the return-to-go: $G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$
>
> $\quad$ Compute the gradient estimate: $\hat{g} = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t$
>
> $\quad$ Update: $\theta \leftarrow \theta + \alpha \hat{g}$

The connection to supervised learning is illuminating. In supervised classification, the loss is $-\log \pi_\theta(y \mid x)$ — the negative log-probability of the correct label $y$. The gradient pushes $\theta$ to make the correct label more likely. REINFORCE uses the same $\nabla_\theta \log \pi_\theta(a \mid s)$ term, but there are no "correct" labels — the agent doesn't know which action was best. Instead, it weights the gradient by the return $G_t$. Actions that led to high returns get reinforced (made more probable); actions that led to low returns get suppressed. REINFORCE is, in this sense, **supervised learning where the labels are the actions you took, weighted by how well they worked**.

### 3.2 A Worked Example

Consider a simple two-action bandit problem: the agent must choose between action A (expected return 1) and action B (expected return 3). The policy is a softmax over two logits $\theta = [\theta_A, \theta_B]$:

$$\pi_\theta(A) = \frac{e^{\theta_A}}{e^{\theta_A} + e^{\theta_B}}, \quad \pi_\theta(B) = \frac{e^{\theta_B}}{e^{\theta_A} + e^{\theta_B}}$$

```python
import numpy as np

np.random.seed(42)
theta = np.array([0.0, 0.0])  # equal logits → 50/50 policy
alpha = 0.1
true_returns = {"A": 1.0, "B": 3.0}

for step in range(8):
    probs = np.exp(theta) / np.exp(theta).sum()
    action = np.random.choice(["A", "B"], p=probs)
    G = true_returns[action] + np.random.randn() * 0.5  # noisy return

    idx = 0 if action == "A" else 1
    # ∇ log π(a|s) for softmax: e_a - π  (one-hot minus probs)
    grad_log_pi = -probs.copy()
    grad_log_pi[idx] += 1.0

    theta += alpha * grad_log_pi * G
    probs_after = np.exp(theta) / np.exp(theta).sum()
    print(f"Step {step}: chose {action}, G={G:+.2f}, "
          f"π(A)={probs_after[0]:.3f}, π(B)={probs_after[1]:.3f}")
```

Over a few steps, the probability of action B increases as its higher returns reinforce it more strongly. But notice the noise: individual updates can push $\theta$ in the wrong direction when the return is noisy. This is the variance problem.

### 3.3 The Variance Problem

REINFORCE uses the full trajectory return $G_t$ to weight each action's gradient. This makes the gradient estimate **unbiased** — in expectation, it equals the true policy gradient. But the variance can be enormous.

Consider an environment where every trajectory has a positive return (say, between 50 and 100). Even bad trajectories (return 50) produce positive weight, so *every* action gets reinforced. The signal is in the *relative* magnitude of the weight (100 vs. 50), but the absolute magnitude causes large, noisy gradient updates. The agent is trying to detect a difference of 50 against a baseline of 50–100 — a poor signal-to-noise ratio.

The same problem appears in a different guise when episodes are long: the return $G_t$ is a sum of many random terms. Each term adds variance but, for early time steps, most of those random rewards happened long after the action and were influenced by many subsequent decisions. The action at time 0 gets "credit" (or blame) for rewards at time 100, even though those rewards were almost entirely determined by later actions. This is the **temporal credit assignment** problem in its most extreme form.

The consequence is that vanilla REINFORCE needs many trajectories to average out the noise — it is **sample-inefficient**. In the Atari domain, REINFORCE would need millions of episodes to learn what DQN learns in thousands. This motivates the variance reduction techniques in the next section.

---

## 4. Variance Reduction: Baselines and Advantage Functions

The policy gradient theorem gives us an unbiased gradient estimate, but with high variance. The core insight of this section is that we can subtract a **baseline** from the return without changing the expected gradient — but dramatically reducing its variance. This is the most important practical improvement to vanilla REINFORCE and the conceptual bridge to actor-critic methods.

### 4.1 The Baseline Trick

Consider subtracting a state-dependent function $b(s_t)$ from the return in the policy gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left(G_t - b(s_t)\right)\right]$$

**Claim:** this is still an unbiased estimate of $\nabla_\theta J$, for any function $b(s)$ that does not depend on the action $a_t$.

The proof is short. The baseline term contributes:

$$\mathbb{E}_{a_t \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \, b(s_t)\right] = b(s_t) \sum_{a} \nabla_\theta \pi_\theta(a \mid s_t) = b(s_t) \, \nabla_\theta \underbrace{\sum_{a} \pi_\theta(a \mid s_t)}_{= 1} = 0$$

Since $\sum_a \pi_\theta(a \mid s_t) = 1$ for any $\theta$ (probabilities always sum to 1), the gradient of a constant is zero. The baseline term vanishes in expectation, so the estimate remains unbiased regardless of what $b(s)$ we choose.

But different baselines give very different variance. The strictly variance-minimizing baseline can be written as a particular weighted conditional expectation of $G_t$ given $s_t$, involving the score $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ (Williams, 1992; Sutton & Barto, 2018, §13.4). It is unwieldy to estimate in deep RL, so implementations almost always use $b(s_t) = V^\pi(s_t)$ or a learned critic $V_\phi(s_t)$, which is simple and dramatically cuts variance even if not exactly optimal. The intuition is that $G_t - V^\pi(s_t)$ centers the weight around zero: trajectories that are *better than average* (from this state) get positive weight, and trajectories that are *worse than average* get negative weight. This eliminates the problem of all weights being positive.

### 4.2 The Advantage Function

Setting $b(s_t) = V^\pi(s_t)$, the weight in the policy gradient becomes:

$$G_t - V^\pi(s_t) \approx Q^\pi(s_t, a_t) - V^\pi(s_t) \triangleq A^\pi(s_t, a_t)$$

This is the **advantage function**: how much better action $a_t$ was compared to the average action from state $s_t$. Reading it in first person: "I'm in state $s_t$, and on average I expect to get $V^\pi(s_t)$ from here. I took action $a_t$ and got (or expect to get) $Q^\pi(s_t, a_t)$. The advantage tells me: *was this action above or below my average performance from this state?*"

$$\underbrace{A^\pi(s, a)}_{\substack{\text{advantage:} \\ \text{how much better} \\ \text{than average?}}} = \underbrace{Q^\pi(s, a)}_{\substack{\text{value of} \\ \text{this specific action}}} - \underbrace{V^\pi(s)}_{\substack{\text{average value} \\ \text{from this state}}}$$

The policy gradient with the advantage function becomes:

**Result: the policy gradient with advantage baseline**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, A^\pi(s_t, a_t)\right]$$

This says: increase the probability of actions with positive advantage (better than average) and decrease the probability of actions with negative advantage (worse than average). Actions that are exactly average ($A = 0$) produce no gradient signal — the policy leaves them unchanged.

In practice, we don't know $A^\pi$ exactly. We estimate it. The simplest estimate is the **one-step TD advantage**:

$$\hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

where $V_\phi$ is a learned value function (the "critic" — we'll develop this in Section 5). Notice this is exactly the TD error from *RL Foundations*, Section 6.1 — the same quantity that drives Q-learning and TD(0). The one-step TD error is a biased but low-variance estimate of the advantage. It's biased because $V_\phi$ is only an approximation of $V^\pi$, but its variance is low because it uses only one step of randomness (the transition from $s_t$ to $s_{t+1}$), rather than the entire future trajectory.

### 4.3 Generalized Advantage Estimation (GAE)

The one-step TD advantage and the full Monte Carlo return represent two extremes of a bias-variance tradeoff:

| Estimator | Bias | Variance | Formula |
|---|---|---|---|
| Monte Carlo ($G_t - V_\phi(s_t)$) | None (if $V_\phi = V^\pi$) | High | Uses all future rewards |
| One-step TD ($r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$) | Higher (bootstraps off $V_\phi$) | Low | Uses one reward + estimate |

This should feel familiar — it's the same tradeoff that TD($\lambda$) navigates in the value-function setting (*RL Foundations*, Section 6.3). Just as TD($\lambda$) blends $n$-step value targets using a decay parameter $\lambda$, **Generalized Advantage Estimation** (GAE; Schulman et al., 2016) blends $n$-step advantage estimates.

Define the one-step TD error (the individual "advantage signal" at each time step):

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

Then the GAE estimator is an exponentially weighted sum of these TD errors:

**Result: Generalized Advantage Estimation (GAE)**

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{\ell=0}^{T-1-t} (\gamma \lambda)^\ell \, \delta_{t+\ell}$$

$$= \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

The parameter $\lambda \in [0, 1]$ controls the tradeoff:

- **$\lambda = 0$:** $\hat{A}_t = \delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. One-step TD advantage. Low variance, higher bias (relies heavily on $V_\phi$).
- **$\lambda = 1$:** $\hat{A}_t = G_t - V_\phi(s_t)$. Full Monte Carlo advantage (same as REINFORCE with baseline). No bias (given correct $V_\phi$), high variance.
- **$\lambda \in (0, 1)$:** A smooth blend. In practice, $\lambda = 0.95$–$0.97$ is common — close to Monte Carlo, but with enough bias to tame the variance.

The computation is efficient — just a backward pass through the TD errors:

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE advantages from a batch of rewards and value estimates.

    `values` should have length T+1: one value per timestep plus a bootstrap
    value V(s_{T+1}) at the end.  If the episode terminated, set values[-1]=0.
    If the batch ended mid-episode, set values[-1]=V_phi(s_{T+1}).
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages
```

The loop runs backward from the end of the episode, accumulating the discounted TD errors. This is the exact same structure as eligibility traces in TD($\lambda$) — and for the same reason: information from the future (which TD errors were large) must propagate backward to earlier time steps.

GAE is used by virtually every modern policy gradient algorithm, including PPO and SAC. It is the standard way to estimate advantages in practice.

---

## 5. Actor-Critic Methods

REINFORCE with a baseline already hints at a two-component structure: the policy (which selects actions) and the value function (which provides the baseline). **Actor-critic** methods make this structure explicit and reap a fundamental benefit: the agent can learn from every single time step, rather than waiting for complete episodes.

### 5.1 The Core Idea: Don't Wait for the Episode to End

REINFORCE is a **Monte Carlo** method — it must complete an entire episode to compute $G_t$, then uses those returns to update the policy. This is wasteful: if the agent is halfway through a 1000-step episode, it has already observed 500 transitions that carry useful information, but REINFORCE ignores them until the episode terminates.

Actor-critic methods fix this by replacing $G_t$ with a **bootstrapped estimate** — the TD error or GAE advantage — computed from a learned value function. The two components are:

- **Actor** ($\pi_\theta$): the policy network. It decides which action to take. Updated by policy gradient, using the critic's advantage estimate as the weight.
- **Critic** ($V_\phi$): the value network. It estimates $V^\pi(s)$ — how good each state is under the current policy. Updated by minimizing TD error, exactly like the value function in TD(0) (*RL Foundations*, Section 6.1).

The actor uses the critic's output to compute advantages; the critic uses the actor's trajectory data to improve its value estimates. They bootstrap off each other.

[FIG:ORIGINAL — Actor-critic architecture diagram showing the interaction between the actor (policy network π_θ) and critic (value network V_ϕ). The environment sends states to both; the actor outputs actions; the critic outputs V(s); the TD error δ = r + γV(s') − V(s) is computed and used to update both networks. Arrows should show the data flow: environment → state → actor → action → environment → reward → critic → advantage → actor update.]

This architecture has a direct ancestor: the **Adaptive Heuristic Critic** (AHC) of Barto, Sutton & Anderson (1983), discussed in *RL Foundations*, Section 6.2. The AHC used a critic that learned $V(s)$ via TD(0) and an actor that used the TD error as a reward signal to update the policy. Modern actor-critic methods follow the same blueprint, scaled up with neural networks and more sophisticated advantage estimation.

### 5.2 A2C: Advantage Actor-Critic

The **Advantage Actor-Critic** (A2C) is the synchronous, single-agent version of the actor-critic framework. It is the clearest expression of the idea and the best starting point.

**Algorithm: A2C (single-environment version)**

> Initialize actor parameters $\theta$ and critic parameters $\phi$.
>
> Repeat:
>
> $\quad$ Collect a batch of $T$ transitions $(s_t, a_t, r_t, s_{t+1})$ by following $\pi_\theta$.
>
> $\quad$ Compute TD errors: $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$
>
> $\quad$ Compute advantages $\hat{A}_t$ using GAE (or just $\delta_t$ for one-step).
>
> $\quad$ **Actor update** (policy gradient with advantage):
>
> $$L_\text{actor}(\theta) = -\frac{1}{T}\sum_{t} \log \pi_\theta(a_t \mid s_t) \, \hat{A}_t$$
>
> $\qquad$ $\theta \leftarrow \theta - \alpha_\theta \nabla_\theta L_\text{actor}$ (gradient descent on the negative = gradient ascent on $J$)
>
> $\quad$ **Critic update** (value function regression):
>
> $$L_\text{critic}(\phi) = \frac{1}{T}\sum_{t} \left(V_\phi(s_t) - \hat{V}_t^{\text{target}}\right)^2$$
>
> $\qquad$ where $\hat{V}_t^{\text{target}} = G_t$ (Monte Carlo target) or $r_t + \gamma V_\phi(s_{t+1})$ (TD target)
>
> $\qquad$ $\phi \leftarrow \phi - \alpha_\phi \nabla_\phi L_\text{critic}$

The actor loss $L_\text{actor}$ is the negative of the policy gradient surrogate — minimizing it is equivalent to maximizing expected return. The negative sign is a convention that lets us use standard gradient descent optimizers (like Adam) for both the actor and the critic.

The critic loss $L_\text{critic}$ is a standard regression loss — it trains $V_\phi$ to predict the expected return from each state. This is the same objective as TD(0) learning (*RL Foundations*, Section 6.1), just expressed as an MSE loss for neural network training.

**Entropy bonus.** A common addition is an entropy term in the actor loss:

$$L_\text{total} = L_\text{actor} - c_H \, H(\pi_\theta(\cdot \mid s))$$

where $H(\pi_\theta(\cdot \mid s)) = -\sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$ is the entropy of the policy (for discrete actions; for continuous Gaussian policies, the differential entropy $H = \frac{1}{2}\log(2\pi e \sigma^2)$ per dimension is used instead). Minimizing $-H$ (i.e., maximizing entropy) discourages the policy from collapsing to a deterministic action too early, promoting exploration. The coefficient $c_H$ controls the strength of this regularization — typically $0.01$–$0.05$.

**Connection to DQN.** The critic in A2C is doing essentially the same job as the Q-network in DQN (*RL in Practice*, Section 4.4) — learning a value function from TD errors. The difference is that A2C's critic learns $V(s)$ (state values) rather than $Q(s, a)$ (action values), and it serves the actor rather than directly determining the policy. DQN has no actor at all — the policy is implicit in the $\arg\max$ over Q-values.

### 5.3 A3C: Asynchronous Advantage Actor-Critic

**A3C** (Mnih et al., 2016) extends A2C by running multiple copies of the agent in parallel, each interacting with its own instance of the environment. The parallel agents asynchronously compute gradients and apply them to a shared parameter set.

The key benefit is **data diversity**: because the agents are in different states at different times, the gradients they contribute are less correlated. This achieves a similar decorrelation effect to experience replay in DQN — but without the replay buffer and without the off-policy complications it introduces. Each worker is on-policy at all times.

In practice, the synchronous variant A2C (which collects batches from all workers, averages the gradients, and applies a single update) often performs as well as or better than the asynchronous A3C, and is simpler to implement and debug. Modern codebases overwhelmingly use A2C or its PPO descendant rather than A3C.

### 5.4 Connection to the Adaptive Heuristic Critic

The Adaptive Heuristic Critic (AHC), introduced in *RL Foundations*, Section 6.2, was the original actor-critic architecture: a critic that learns $V(s)$ via TD(0), and an actor that treats each state as an independent bandit problem, using the TD error as a local reward signal.

Modern actor-critic methods differ in two ways. First, the actor uses the policy gradient theorem rather than independent bandit updates — this accounts for the sequential structure of the problem. Second, both actor and critic are parameterized by neural networks with shared or separate parameters, enabling generalization across states.

But the core insight is identical: the TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is a sample of the advantage — "how much better was this transition than expected?" — and the actor uses it to reinforce good actions and suppress bad ones. The progression from AHC (1983) to A2C (2016) is a remarkably straight line: same idea, better optimization machinery.

---

## 6. Trust Regions: Why Step Size Matters in Policy Space

Actor-critic methods tell us *which direction* to update the policy (the policy gradient). But they don't tell us *how far* to step. In supervised learning, taking a slightly-too-large gradient step usually just overshoots and can be corrected on the next step. In RL, a bad policy update can be catastrophic — and the damage is much harder to undo.

### 6.1 The Policy Collapse Problem

Suppose the current policy is good — it reaches the goal in 20 steps. A gradient update overshoots and makes the policy slightly worse: it now reaches the goal in 50 steps, or doesn't reach it at all. In the next batch, all the data comes from the *new, bad policy*, so the gradient estimate is computed from low-quality trajectories. The next update, based on this bad data, might make the policy even worse. The feedback loop can cascade rapidly, collapsing the policy to random or degenerate behavior.

This failure mode is specific to RL. In supervised learning, the training data is fixed — a bad weight update doesn't change the labels. In RL, the data distribution *is* the policy, so changing the policy changes the data, which changes the gradients, which changes the policy. A single bad step can corrupt the entire learning process.

The problem is that small changes in parameter space $\theta$ can cause large changes in policy behavior. A small weight perturbation in a softmax layer could shift probability mass from one action to another entirely, causing the agent to visit completely different states.

### 6.2 Measuring Policy Change: KL Divergence

To control how much the policy changes per update, we need a measure of "distance" between two policies. The natural choice is the **KL divergence** (Kullback-Leibler divergence), which measures how different two probability distributions are:

$$D_{\text{KL}}(\pi_{\theta_\text{old}} \| \pi_{\theta_\text{new}}) = \mathbb{E}_{s \sim \rho}\left[\sum_a \pi_{\theta_\text{old}}(a \mid s) \log \frac{\pi_{\theta_\text{old}}(a \mid s)}{\pi_{\theta_\text{new}}(a \mid s)}\right]$$

where $\rho$ is the state visitation distribution under the old policy. KL divergence is zero when the two policies are identical and increases as they diverge. Crucially, it measures distance in *policy space* (how different the action distributions are), not in *parameter space* (how different the weight vectors are). A large parameter change that doesn't affect action probabilities has zero KL divergence.

The **natural policy gradient** (Kakade, 2001) uses the Fisher information matrix (the curvature of the KL divergence) to precondition the gradient. Instead of taking a fixed-size step in parameter space, it takes a fixed-size step in policy space — ensuring that each update changes the policy by a controlled amount regardless of the parameterization. This is theoretically elegant but computationally expensive (it requires inverting a large matrix).

### 6.3 TRPO: Trust Region Policy Optimization

**TRPO** (Schulman et al., 2015) operationalizes the trust region idea: maximize the policy improvement subject to a constraint on how much the policy is allowed to change.

The optimization problem is:

$$\max_\theta \; \mathbb{E}_{s, a \sim \pi_{\theta_\text{old}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_\text{old}}(a \mid s)} \hat{A}(s, a)\right] \quad \text{subject to} \quad D_{\text{KL}}(\pi_{\theta_\text{old}} \| \pi_\theta) \leq \delta$$

Reading this in first person: "I want to find new parameters $\theta$ that make good actions (positive advantage) more probable and bad actions (negative advantage) less probable. The ratio $\pi_\theta / \pi_{\theta_\text{old}}$ tells me how much I'm changing each action's probability. But I'm not allowed to change the overall policy too much — the KL divergence must stay within $\delta$."

The ratio $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ is the **importance sampling ratio** — it reweights the advantage from the old policy's data to estimate the new policy's performance. When $r_t = 1$, the new and old policies agree on this action. When $r_t > 1$, the new policy makes this action more likely; when $r_t < 1$, less likely.

TRPO solves this constrained optimization using a second-order approximation: it computes the natural gradient direction via conjugate gradient (avoiding the full Fisher matrix inversion) and then performs a line search to find the largest step that satisfies the KL constraint.

**Why TRPO works:** Kakade & Langford (2002) proved that if the KL divergence between successive policies is bounded by $\delta$, then the policy improvement is guaranteed to be at least:

$$J(\theta_\text{new}) \geq J(\theta_\text{old}) + \underbrace{\mathbb{E}_{s \sim \rho_{\theta_\text{old}}}\left[\mathbb{E}_{a \sim \pi_{\theta_\text{new}}}[A^{\pi_{\theta_\text{old}}}(s, a)]\right]}_{\text{surrogate advantage (how much better the new policy looks)}} - C\delta$$

where $C$ is a constant depending on $\gamma$ and the advantage magnitude. As long as the expected advantage exceeds $C\delta$, the update is guaranteed to improve the policy. This **monotonic improvement guarantee** is TRPO's theoretical contribution — it converts the uncontrolled policy gradient into a procedure with provable progress. The bound is stated for the **idealized** surrogate-and-KL setup; practical TRPO uses quadratic trust-region approximations, a finite number of conjugate-gradient steps, and line search, so the inequality is **motivational** — a design principle — rather than a literal certificate after every update.

**Why TRPO is hard to use:** The conjugate gradient computation and line search make TRPO substantially more complex to implement than vanilla policy gradients. It is also incompatible with architectures that share parameters between the actor and critic (common for efficiency), because the KL constraint applies to the actor but the shared parameters also affect the critic. These practical difficulties motivated PPO.

---

## 7. PPO: Proximal Policy Optimization

PPO (Schulman et al., 2017) achieves the same goal as TRPO — preventing destructively large policy updates — but with a much simpler mechanism. Instead of solving a constrained optimization problem with conjugate gradients and line search, PPO uses a **clipped surrogate objective** that automatically limits the effective step size. It is the most widely used policy gradient algorithm in practice.

### 7.1 The Clipped Surrogate Objective

Start with the TRPO surrogate objective (the ratio times the advantage):

$$L^{\text{CPI}}(\theta) = \mathbb{E}_t\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)} \hat{A}_t\right] = \mathbb{E}_t\left[r_t(\theta) \, \hat{A}_t\right]$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ is the probability ratio. Without any constraint, maximizing this objective would encourage arbitrarily large changes to the policy.

PPO clips the ratio to prevent it from moving too far from 1:

**Result: the PPO clipped surrogate objective**

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \, \hat{A}_t, \; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon) \, \hat{A}_t\right)\right]$$

where $\epsilon$ is a small hyperparameter (typically $0.1$–$0.2$). The $\text{clip}$ function restricts the ratio to the interval $[1 - \epsilon, 1 + \epsilon]$.

The $\min$ takes the more pessimistic of two terms: the unclipped objective and the clipped objective. To understand why, consider two cases:

**Case 1: Positive advantage ($\hat{A}_t > 0$).** The action was better than average. The unclipped objective $r_t \hat{A}_t$ wants to increase $r_t$ (make this good action more likely). But the clipped version caps the benefit at $(1 + \epsilon)\hat{A}_t$. Once $r_t > 1 + \epsilon$, further increases don't increase the objective — the gradient is zero. "This action was good, so I want to make it more likely — but not *too much* more likely."

**Case 2: Negative advantage ($\hat{A}_t < 0$).** The action was worse than average. The unclipped objective wants to decrease $r_t$ (make this bad action less likely). But the clipped version caps the penalty at $(1 - \epsilon)\hat{A}_t$. Once $r_t < 1 - \epsilon$, the gradient is zero. "This action was bad, so I want to make it less likely — but not *too much* less likely."

In both cases, the clipping creates a "trust region" around the old policy: the update can change action probabilities by at most a factor of $1 \pm \epsilon$, then it stops. This prevents the catastrophic policy collapse described in Section 6.1 — and it does so with a single line of code, no conjugate gradients required.

[FIG:ORIGINAL — PPO clipped surrogate objective plotted as a function of the probability ratio r_t(θ), showing two panels side by side: one for positive advantage (A > 0) where the objective is linear up to r = 1+ε then flat, and one for negative advantage (A < 0) where the objective is linear down to r = 1−ε then flat. The flat regions (zero gradient) should be shaded to show where clipping prevents further updates. This is the standard PPO clipping visualization from Schulman et al. 2017 or OpenAI Spinning Up.]

```python
# PPO clipped objective (PyTorch-style pseudocode)
# Compute the ratio in log space for numerical stability:
# exp(log π_new - log π_old) instead of exp(log π_new) / exp(log π_old)
ratio = (new_log_probs - old_log_probs).exp()        # r_t(θ)
clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
```

### 7.2 The Full Algorithm

**Algorithm: PPO (Clip variant)**

> Initialize actor $\pi_\theta$ and critic $V_\phi$.
>
> Repeat:
>
> $\quad$ **Collect data.** Run $\pi_{\theta_\text{old}}$ for $T$ timesteps across $N$ parallel environments, storing $(s_t, a_t, r_t, s_{t+1}, \log \pi_{\theta_\text{old}}(a_t \mid s_t))$.
>
> $\quad$ **Compute advantages.** Use GAE with the critic $V_\phi$ to compute $\hat{A}_t$ for each timestep.
>
> $\quad$ **Optimize.** For $K$ epochs (typically 3–10), on random minibatches of the collected data:
>
> $\qquad$ Compute $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$
>
> $\qquad$ Actor loss: $L_\text{actor} = -\mathbb{E}_t\left[\min\left(r_t \hat{A}_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$
>
> $\qquad$ Critic loss: $L_\text{critic} = \mathbb{E}_t\left[\left(V_\phi(s_t) - \hat{V}_t^{\text{target}}\right)^2\right]$
>
> $\qquad$ Total loss: $L = L_\text{actor} + c_v L_\text{critic} - c_H H(\pi_\theta)$
>
> $\qquad$ Update $\theta$ and $\phi$ by gradient descent on $L$.
>
> $\quad$ Set $\theta_\text{old} \leftarrow \theta$.

Two features distinguish PPO from vanilla A2C:

1. **Multiple epochs on the same data.** PPO reuses each batch of collected data for $K$ optimization steps. In vanilla policy gradients, data is used once and discarded. The clipping mechanism makes this safe: even after several gradient steps, the ratio $r_t$ cannot drift too far from 1, so the data remains approximately valid.

2. **Stored log-probabilities.** The old policy's log-probabilities are stored during data collection. This is necessary to compute the ratio $r_t$ during optimization, when $\theta$ has already changed from $\theta_\text{old}$.

**Typical hyperparameters:**

| Parameter | Typical Value | Role |
|---|---|---|
| $\epsilon$ (clip range) | 0.1–0.2 | How far the policy can change per update |
| $K$ (epochs per batch) | 3–10 | How many times to reuse each data batch |
| $\lambda$ (GAE) | 0.95 | Bias-variance tradeoff for advantages |
| $c_v$ (critic coefficient) | 0.5 | Weight of critic loss in total loss |
| $c_H$ (entropy coefficient) | 0.01 | Exploration incentive |
| Minibatch size | 64–4096 | Per-update sample size |
| Number of parallel envs | 8–256 | Data collection throughput |

### 7.3 Why PPO Became the Default

PPO achieved something rare: strong theoretical motivation (inherited from TRPO) with extreme practical simplicity. The algorithm can be implemented in roughly 100 lines of PyTorch, requires no second-order optimization, and works well across a wide range of tasks with minimal hyperparameter tuning.

PPO is the algorithm behind:

- **OpenAI Five** (Berner et al., 2019): defeated the world champion Dota 2 team using massively distributed PPO.
- **InstructGPT / ChatGPT** (Ouyang et al., 2022): used PPO to fine-tune language models from human feedback (Section 10.1).
- **OpenAI Gym baselines**: PPO is the default algorithm for continuous control benchmarks (MuJoCo, Atari).

Its dominance stems from reliability: while other algorithms (SAC, TD3) can achieve higher asymptotic performance on specific tasks, PPO is the algorithm most likely to produce reasonable results on a new task with default hyperparameters. In RL, where debugging is notoriously difficult, this robustness is invaluable.

---

## 8. Deterministic Policy Gradients and DDPG

The policy gradient methods developed so far use stochastic policies — the network outputs a probability distribution and actions are sampled from it. But for continuous control tasks, there's an alternative: learn a **deterministic** policy $\mu_\theta(s)$ that directly outputs the action to take in each state, with no randomness.

### 8.1 The DPG Theorem

Silver et al. (2014) proved that deterministic policies have their own gradient theorem. For a deterministic policy $a = \mu_\theta(s)$, the policy gradient is:

**Result: the Deterministic Policy Gradient (DPG)**

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu}\left[\nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s, a)\big|_{a = \mu_\theta(s)}\right]$$

where $\rho^\mu$ is the **discounted state visitation distribution** under policy $\mu$ — the probability of being in each state, weighted by how often the agent visits it during rollouts. Reading this: "For each state I visit, I compute how my action output changes with my parameters ($\nabla_\theta \mu_\theta$) and how the Q-value changes with the action ($\nabla_a Q^\mu$). I chain them together to get the direction in parameter space that most increases the Q-value of my chosen actions."

The key difference from the stochastic policy gradient: the DPG does not require sampling actions — it uses the deterministic action $\mu_\theta(s)$ directly. And it requires the gradient of the Q-function with respect to actions, $\nabla_a Q$, which means we need a differentiable Q-function approximator (a neural network critic).

The advantage is **lower variance**: the stochastic policy gradient integrates over the action distribution (which adds noise from sampling), while the DPG only integrates over the state distribution. This makes DPG more sample-efficient for continuous action spaces.

The disadvantage is **no exploration by default**: a deterministic policy always takes the same action in the same state. Exploration must be added externally.

### 8.2 DDPG: Deep Deterministic Policy Gradient

**DDPG** (Lillicrap et al., 2016) scales the DPG to deep neural networks by borrowing two stabilization techniques from DQN (*RL in Practice*, Section 4.4):

- **Experience replay buffer:** stores transitions $(s, a, r, s')$ and samples random minibatches for training, breaking temporal correlations — exactly as in DQN.
- **Target networks:** maintains slowly-updated copies of both the actor and critic ($\mu_{\theta'}$, $Q_{\phi'}$) for computing TD targets, preventing the moving-target problem. Unlike DQN, which uses **hard** updates (periodic full copy every $C$ steps), DDPG uses **soft** updates (**Polyak averaging**) after every gradient step: $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$, with $\tau \approx 0.005$. This gives smoother, more stable target evolution — each update blends 0.5% of the new weights into the target, rather than replacing them wholesale.

The actor and critic are updated as:

**Critic update:** Minimize the Bellman error, using the target networks for the TD target:

$$y_t = r_t + \gamma \, Q_{\phi'}(s_{t+1}, \mu_{\theta'}(s_{t+1}))$$

$$L_\text{critic}(\phi) = \mathbb{E}\left[(Q_\phi(s_t, a_t) - y_t)^2\right]$$

**Actor update:** Maximize $Q_\phi$ by applying the DPG:

$$\nabla_\theta J \approx \mathbb{E}\left[\nabla_\theta \mu_\theta(s) \, \nabla_a Q_\phi(s, a)\big|_{a=\mu_\theta(s)}\right]$$

**Exploration:** DDPG adds noise directly to the deterministic action: $a = \mu_\theta(s) + \mathcal{N}(0, \sigma^2)$, where $\sigma$ is decayed over training. Early work used Ornstein-Uhlenbeck noise (temporally correlated), but simple Gaussian noise works equally well.

DDPG can be thought of as "DQN for continuous actions" — it uses the same replay buffer and target network infrastructure, but replaces the discrete $\arg\max$ with a differentiable actor network. This connection is why it's often introduced alongside DQN in the function approximation context.

**Limitations.** DDPG can be brittle: it is sensitive to hyperparameters, can overestimate Q-values (the same maximization bias from *RL Foundations*, Section 5.9), and sometimes fails to explore adequately. **TD3** (Twin Delayed DDPG; Fujimoto et al., 2018) addresses these issues with three modifications: clipped double Q-learning (two critics, take the minimum), delayed actor updates (update the critic more frequently), and target policy smoothing (add noise to the target action). We do not cover TD3 in detail, but its ideas are subsumed by SAC.

---

## 9. SAC: Soft Actor-Critic

Soft Actor-Critic (Haarnoja et al., 2018) takes a different philosophical stance: instead of maximizing return alone, the agent maximizes return *plus* entropy. This seemingly small change has profound consequences for exploration, robustness, and training stability.

### 9.1 Maximum Entropy Reinforcement Learning

Standard RL maximizes the expected cumulative reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

**Maximum entropy RL** adds an entropy bonus at every time step:

$$J_\text{soft}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t \Big(r_t + \alpha \, H\big(\pi_\theta(\cdot \mid s_t)\big)\Big)\right]$$

where $H(\pi_\theta(\cdot \mid s)) = -\sum_a \pi_\theta(a \mid s) \log \pi_\theta(a \mid s)$ is the entropy of the policy in state $s$, and $\alpha > 0$ is the **temperature** parameter controlling the tradeoff.

Reading this in first person: "I want to maximize my reward, *but* I also want to stay as random as possible. $\alpha$ controls how much I value randomness relative to reward. When $\alpha$ is large, I prefer to keep my options open; when $\alpha$ is small, I focus on reward."

Why add entropy? Three reasons:

1. **Exploration.** The entropy bonus discourages premature convergence to a deterministic policy. Even late in training, the agent maintains some randomness, continuing to explore. This is especially valuable in environments with multiple near-optimal strategies — standard RL might lock onto one and never discover the others.

2. **Robustness.** A stochastic policy that performs well is inherently more robust than a deterministic one: it has learned a distribution of good behaviors rather than a single brittle strategy. When the environment changes slightly, the stochastic policy is more likely to still contain a good response.

3. **Composability.** Maximum-entropy policies can be composed more easily for hierarchical tasks (Haarnoja et al., 2018). Intuitively, a policy that maximizes entropy at every state provides a richer set of "primitive behaviors" to build on.

The **soft value functions** are defined analogously to their standard counterparts, but with entropy included:

$$V_\text{soft}^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t (r_t + \alpha \, H(\pi(\cdot \mid s_t))) \;\middle|\; s_0 = s\right]$$

$$Q_\text{soft}^\pi(s, a) = r(s, a) + \gamma \, \mathbb{E}_{s'}\left[V_\text{soft}^\pi(s')\right]$$

$$V_\text{soft}^\pi(s) = \mathbb{E}_{a \sim \pi}\left[Q_\text{soft}^\pi(s, a) - \alpha \log \pi(a \mid s)\right]$$

The last equation is the **soft Bellman equation**: the value of a state includes not just the expected Q-value but also the entropy of the action distribution. Higher entropy (more randomness) directly increases the state value.

### 9.2 The SAC Architecture

SAC maintains five network instances:

1. **Two online Q-networks** $Q_{\phi_1}(s, a)$ and $Q_{\phi_2}(s, a)$: Two independent critics, trained on the same data. Taking the minimum of the two (clipped double Q-learning, as in TD3) prevents overestimation.

2. **Two target Q-networks** $Q_{\phi'_1}$ and $Q_{\phi'_2}$: Slowly-updated copies of the online critics, using Polyak averaging ($\phi' \leftarrow \tau\phi + (1-\tau)\phi'$) just as in DDPG. These provide stable TD targets for the critic update.

3. **Policy network** $\pi_\theta(a \mid s)$: Outputs the parameters of a squashed Gaussian distribution. For a continuous action space, the network outputs mean $\mu_\theta(s)$ and log-standard-deviation $\log \sigma_\theta(s)$.

**The reparameterization trick.** The actor update requires backpropagating through a sampled action $\tilde{a} \sim \pi_\theta$ — but sampling is not a differentiable operation. SAC uses the **reparameterization trick** to move the randomness out of the computation graph: instead of "sample $\tilde{a}$ from $\pi_\theta$", write:

$$\tilde{a} = \tanh\!\big(\mu_\theta(s) + \sigma_\theta(s) \odot \epsilon\big), \quad \epsilon \sim \mathcal{N}(0, I)$$

Now $\epsilon$ is fixed external noise (not a function of $\theta$), so gradients flow through $\mu_\theta$ and $\sigma_\theta$ via standard backpropagation. The $\tanh$ squashing keeps actions within bounds $[-1, 1]$. Because squashing changes the density, the log-probability requires a correction via the change-of-variables formula:

$$\log \pi_\theta(\tilde{a} \mid s) = \log \mathcal{N}(u; \mu, \sigma^2) - \sum_{i=1}^{d} \log(1 - \tanh^2(u_i))$$

where $u = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon$ is the pre-squashing sample and $d$ is the action dimensionality. The second term accounts for the volume change introduced by $\tanh$.

The updates are:

**Critic update:** Minimize the soft Bellman error:

$$y_t = r_t + \gamma \left[\min(Q_{\phi'_1}(s', \tilde{a}'),\, Q_{\phi'_2}(s', \tilde{a}')) - \alpha \log \pi_\theta(\tilde{a}' \mid s')\right]$$

where $\tilde{a}' \sim \pi_\theta(\cdot \mid s')$ is a fresh sample from the current policy. Both critics are updated to minimize $(Q_{\phi_i}(s, a) - y_t)^2$.

**Actor update:** Minimize the expected KL divergence between the policy and the exponential of the soft Q-function:

$$L_\text{actor}(\theta) = \mathbb{E}_{s,\, \tilde{a} \sim \pi_\theta}\left[\alpha \log \pi_\theta(\tilde{a} \mid s) - \min(Q_{\phi_1}(s, \tilde{a}),\, Q_{\phi_2}(s, \tilde{a}))\right]$$

Reading this: "I want to increase the probability of actions that have high Q-value (negative of the Q-terms), but I also want to keep the policy's entropy high (negative of the $\alpha \log \pi$ term). The temperature $\alpha$ balances these two forces."

SAC uses **off-policy** learning with an experience replay buffer, like DQN and DDPG. The **critics** are the main off-policy component: Bellman targets for $Q_\phi$ can use $(s, a, r, s')$ tuples collected under older behavior policies, just as in DQN. The **actor** still depends on the **current** policy (reparameterized samples and entropy terms at replayed states), so the overall algorithm is not “pure replay of independent targets” in the same sense as fitted Q-iteration — but reusing past transitions for the Q-updates is what yields the large sample-efficiency gain over on-policy methods like PPO, which discard data after each policy update.

### 9.3 Automatic Temperature Tuning

The temperature $\alpha$ controls the entropy-reward tradeoff and is hard to set by hand — too high and the agent never exploits; too low and it never explores.

Haarnoja et al. (2018b) proposed **automatic temperature tuning**: treat $\alpha$ as a Lagrange multiplier for a constrained optimization problem. The constraint is that the policy's entropy should stay above a target $\bar{H}$ (typically set to $-\text{dim}(\mathcal{A})$, the negative of the action space dimensionality):

$$\min_\alpha \; \mathbb{E}_{s \sim \mathcal{D}}\left[-\alpha \left(\log \pi_\theta(a \mid s) + \bar{H}\right)\right]$$

When the policy's entropy is above the target, $\alpha$ decreases (less entropy bonus needed). When the entropy drops below the target, $\alpha$ increases (push the policy to be more random). This eliminates a critical hyperparameter and makes SAC more robust across tasks.

### 9.4 Why SAC Matters

SAC achieves state-of-the-art sample efficiency on continuous control benchmarks (MuJoCo locomotion tasks like HalfCheetah, Ant, Humanoid). Compared to its competitors:

| | PPO | DDPG/TD3 | SAC |
|---|---|---|---|
| **On/off-policy** | On-policy | Off-policy | Off-policy |
| **Sample efficiency** | Lower (discards data) | Higher (replay buffer) | Higher (replay buffer) |
| **Stability** | Very stable (clipping) | Can be brittle | Stable (entropy + double Q) |
| **Exploration** | Entropy bonus in the objective (and stochastic policy sampling) | External noise | Built-in (max entropy) |
| **Continuous actions** | Gaussian policy | Deterministic + noise | Squashed Gaussian |
| **Hyperparameter sensitivity** | Low | High | Low (auto-$\alpha$) |

SAC is often the first choice for continuous control tasks where sample efficiency matters (e.g., robotics with expensive real-world data). PPO remains preferred when simplicity and reliability across diverse task types are priorities, or when the task is naturally on-policy (e.g., RLHF, as discussed in Section 10.1).

---

## 10. Connections to Modern Applications

The algorithms in this document — particularly PPO and SAC — are not just academic exercises. They power some of the most impactful applications of machine learning in the 2020s.

### 10.1 RLHF: Reinforcement Learning from Human Feedback

The most widely known application of policy gradients is **RLHF** — the technique used to align large language models (LLMs) with human preferences. InstructGPT (Ouyang et al., 2022) and its descendants (ChatGPT, Claude, Gemini) all use variants of this approach.

The RLHF pipeline has three stages:

[FIG:ORIGINAL — RLHF pipeline diagram showing three stages left to right: (1) Supervised Fine-Tuning (SFT) with a pretrained LLM and demonstration data producing an SFT model, (2) Reward Model Training with human preference comparisons producing a reward model R_ψ, and (3) PPO Fine-Tuning with the SFT model as π_ref, the reward model providing rewards, and a KL penalty constraining the updated policy π_θ. Arrows should show data flow between stages.]

**Stage 1: Supervised fine-tuning (SFT).** Start with a pretrained language model and fine-tune it on a dataset of human-written demonstrations (prompt-response pairs). This gives the model basic instruction-following ability.

**Stage 2: Reward model training.** Collect human preference data: given a prompt, generate two responses and ask a human annotator which is better. Train a **reward model** $R_\psi(x, y)$ — a neural network that takes a prompt $x$ and response $y$ and outputs a scalar score predicting human preference.

**Stage 3: PPO fine-tuning.** Treat the language model as a policy $\pi_\theta$ that generates responses (actions) given prompts (states). The reward model provides the reward signal. Use PPO to optimize $\pi_\theta$ to maximize the reward model's score — with an important constraint:

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\left[R_\psi(x, y) - \beta \, D_{\text{KL}}(\pi_\theta(\cdot \mid x) \| \pi_\text{ref}(\cdot \mid x))\right]$$

The **KL penalty** $\beta \, D_\text{KL}(\pi_\theta \| \pi_\text{ref})$ prevents the policy from drifting too far from the reference model $\pi_\text{ref}$ (the SFT model). Without this constraint, the model would exploit the reward model — finding adversarial outputs that score highly but are nonsensical. This is the RL analog of **reward hacking** (discussed in *RL in Practice*, Section 3.2, Note on reward design): the agent perfectly optimizes the reward function you gave it, which turns out not to capture what you actually wanted.

The connection to the algorithms in this document is direct: the LLM is the actor, the reward model is the environment (one-step, no transitions), the KL penalty acts as a trust region (analogous to TRPO/PPO's clipping), and the optimization is performed by PPO with GAE advantages. The entire RLHF pipeline rests on the policy gradient theorem derived in Section 2.

### 10.2 Other Frontiers

Policy gradient methods form the backbone of several other active research areas:

**Multi-agent RL.** When multiple agents interact (competitive games, cooperative robotics), each agent's environment is non-stationary (because the other agents are learning too). Policy gradient methods are preferred here because they naturally handle stochastic policies (needed for game-theoretic equilibria) and are more robust to non-stationarity than value-based methods that assume fixed transition dynamics.

**Offline RL.** Learning from a fixed dataset of pre-collected transitions, with no further environment interaction. Offline variants of SAC and actor-critic methods (e.g., Conservative Q-Learning, CQL; Kumar et al., 2020) constrain the policy to stay close to the data-collection policy, avoiding out-of-distribution actions.

**World models and model-based policy gradients.** Learning a differentiable model of the environment and backpropagating through it to compute policy gradients — a model-based extension of the methods in this document. This is the idea behind Dreamer (Hafner et al., 2020) and related approaches. The connection to Dyna (*RL in Practice*, Section 3.2) is clear: both learn a model and use it for planning, but Dyna plans with simulated Q-learning updates while world models plan with direct gradient computation through the learned dynamics.

---

## 11. The Full RL Landscape

### 11.1 How the Three Documents Connect

The three companion documents trace the full arc of reinforcement learning:

**Document 1: *RL Foundations*** built the mathematical framework. MDPs, Bellman equations, value functions, and the core algorithms (VI, PI, Q-learning, SARSA, TD($\lambda$)) for solving them. The setting was tabular — each state has its own entry in a lookup table — and actions were discrete.

**Document 2: *RL in Practice*** scaled those ideas to real-world problems. Exploration strategies (bandits, $\epsilon$-greedy, UCB), model-based methods (Dyna, prioritized sweeping), and function approximation (neural networks, DQN, Rainbow) addressed the practical obstacles of large state spaces, sample efficiency, and stable learning.

**Document 3: *Policy Gradients* (this document)** opened the second major branch of RL. Instead of learning value functions and deriving policies, we parameterize policies directly and optimize them by gradient ascent. This enables continuous action spaces, stochastic policies, and a different set of algorithmic tradeoffs (variance reduction, trust regions, maximum entropy).

The progression across all three:

```
Document 1 (Foundations)          Document 2 (Practice)              Document 3 (Policy Gradients)
─────────────────────           ─────────────────────              ─────────────────────────────
MDPs, Bellman eqs  ──────────→  Exploration strategies
                                Model-based (Dyna, PriSweep)
VI, PI (known model) ────────→  Function approx + DQN + Rainbow
                                                                    
Q-learning, SARSA  ──────────→  ├── Value-based scaling ──────→   ├── Policy gradient theorem
TD(λ), eligibility ──────────→  │   (handles large states)        │   REINFORCE, baselines
                                │                                  │   Variance reduction (GAE)
                                │                                  │   Actor-critic (A2C)
                                │                                  │   Trust regions → PPO
                                │                                  │   DPG → DDPG
                                │                                  │   Max entropy → SAC
                                │                                  │   RLHF applications
                                └── Practical obstacles            └── Continuous/stochastic actions
```

The two branches — value-based (Q-learning → DQN → Rainbow) and policy-based (REINFORCE → A2C → PPO/SAC) — share the same theoretical roots (Bellman equations, TD learning, advantage functions) but diverge in how they use those ideas. Actor-critic methods like PPO and SAC explicitly bridge the two branches: they use value functions (the critic) in service of policy optimization (the actor).

### 11.2 Decision Flowchart: Which Algorithm Should I Use?

The choice of algorithm depends on the problem structure. Here is a practical decision guide:

| Situation | Recommended Algorithm | Why |
|---|---|---|
| Small discrete state/action space, known model | **Value Iteration** or **Policy Iteration** | Exact solution, no learning needed |
| Small discrete state/action space, unknown model | **Tabular Q-learning** or **SARSA** | Converges to optimal, simple to implement |
| Need sample efficiency with discrete actions | **Dyna** or **Prioritized Sweeping** | Model-based planning amplifies each experience |
| Large/continuous state space, discrete actions | **DQN** (+ Double, Dueling, PER) | Function approximation with proven stabilization |
| Continuous action space, need reliability | **PPO** | Robust, simple, works across domains |
| Continuous action space, need sample efficiency | **SAC** | Off-policy + entropy = efficient + stable |
| Deterministic continuous control with replay | **TD3** (or **DDPG**) | Deterministic policy + double Q |
| Fine-tuning LLMs from human feedback | **PPO** (with KL penalty) | On-policy stability, compatible with LLM training |
| Safety-critical, online performance matters | **SARSA** or **on-policy actor-critic** | Learns value of actual behavior, not ideal behavior |
| Multi-agent or game-theoretic setting | **PPO** or policy gradient variants | Stochastic policies needed for equilibria |

The overarching pattern: **start with the simplest algorithm that matches your problem's structure**. Tabular methods for small problems, DQN for large discrete problems, PPO for anything with continuous actions or when you want reliability, and SAC when you need sample efficiency on continuous control tasks.

---

## Sources and Further Reading

### Foundational policy gradient theory

- **Williams, R. J.** (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*, 8, 229–256. — Introduced the REINFORCE algorithm and the score function estimator for policy gradients.
- **Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y.** (1999). Policy Gradient Methods for Reinforcement Learning with Function Approximation. *Advances in Neural Information Processing Systems*, 12. — Proved the policy gradient theorem for parameterized policies with function approximation.
- **Kakade, S.** (2001). A Natural Policy Gradient. *Advances in Neural Information Processing Systems*, 14. — Introduced the natural policy gradient using the Fisher information matrix.
- **Kakade, S. & Langford, J.** (2002). Approximately Optimal Approximate Reinforcement Learning. *Proceedings of ICML*. — Monotonic improvement bounds for conservative policy updates, the theoretical foundation for TRPO.

### Actor-critic and advantage estimation

- **Barto, A. G., Sutton, R. S., & Anderson, C. W.** (1983). Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-13(5), 834–846. — The original actor-critic architecture (Adaptive Heuristic Critic).
- **Konda, V. R. & Tsitsiklis, J. N.** (2000). Actor-Critic Algorithms. *Advances in Neural Information Processing Systems*, 12. — Convergence theory for actor-critic methods with linear function approximation.
- **Mnih, V. et al.** (2016). Asynchronous Methods for Deep Reinforcement Learning. *Proceedings of ICML*. — Introduced A3C (Asynchronous Advantage Actor-Critic) and demonstrated that parallel actors can replace experience replay.
- **Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P.** (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *Proceedings of ICLR*. — Introduced GAE, the standard advantage estimator used in PPO and most modern policy gradient methods.

### Trust regions and proximal methods

- **Schulman, J., Levine, S., Abbeel, P., Jordan, M. I., & Moritz, P.** (2015). Trust Region Policy Optimization. *Proceedings of ICML*. — Introduced TRPO with monotonic improvement guarantees via KL-constrained optimization.
- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*. — Introduced PPO's clipped surrogate objective, the dominant policy gradient algorithm in practice.

### Deterministic and continuous-action methods

- **Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M.** (2014). Deterministic Policy Gradient Algorithms. *Proceedings of ICML*. — Proved the deterministic policy gradient theorem for continuous actions.
- **Lillicrap, T. P. et al.** (2016). Continuous Control with Deep Reinforcement Learning. *Proceedings of ICLR*. — Introduced DDPG, scaling deterministic policy gradients with deep networks, replay buffers, and target networks.
- **Fujimoto, S., van Hoof, H., & Meger, D.** (2018). Addressing Function Approximation Error in Actor-Critic Methods. *Proceedings of ICML*. — Introduced TD3 (Twin Delayed DDPG) with clipped double Q-learning, delayed updates, and target smoothing.

### Maximum entropy RL

- **Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K.** (2008). Maximum Entropy Inverse Reinforcement Learning. *Proceedings of AAAI*. — Introduced the maximum entropy framework for IRL, foundational to the entropy-regularized RL paradigm.
- **Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S.** (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *Proceedings of ICML*. — Introduced SAC with the soft Bellman equation and entropy-regularized objectives.
- **Haarnoja, T. et al.** (2018). Soft Actor-Critic Algorithms and Applications. *arXiv:1812.05905*. — Extended SAC with automatic temperature tuning and demonstrated state-of-the-art continuous control performance.

### Applications and modern extensions

- **Ouyang, L. et al.** (2022). Training Language Models to Follow Instructions with Human Feedback. *Advances in Neural Information Processing Systems*, 35. — Introduced InstructGPT and the RLHF pipeline using PPO for language model alignment.
- **Berner, C. et al.** (2019). Dota 2 with Large Scale Deep Reinforcement Learning. *arXiv:1912.06680*. — Used massively distributed PPO to train OpenAI Five, defeating the world champion Dota 2 team.
- **Kumar, A., Zhou, A., Tucker, G., & Levine, S.** (2020). Conservative Q-Learning for Offline Reinforcement Learning. *Advances in Neural Information Processing Systems*, 33. — Introduced CQL for learning from fixed datasets without environment interaction.
- **Hafner, D. et al.** (2020). Dream to Control: Learning Behaviors by Latent Imagination. *Proceedings of ICLR*. — World model approach combining learned dynamics with actor-critic policy optimization.

### Textbooks and surveys

- **Sutton, R. S. & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. — Chapter 13 covers policy gradient methods; Chapter 15 covers applications. The standard reference for the field.
- **Levine, S.** (2020). CS 285: Deep Reinforcement Learning. UC Berkeley. — Lecture notes covering policy gradients, actor-critic, TRPO, PPO, SAC, and model-based RL. Available at [rail.eecs.berkeley.edu/deeprlcourse/](http://rail.eecs.berkeley.edu/deeprlcourse/).
- **Achiam, J.** (2018). Spinning Up in Deep RL. OpenAI. — Practical introduction to implementing PPO, SAC, DDPG, and TD3. Available at [spinningup.openai.com](https://spinningup.openai.com).
