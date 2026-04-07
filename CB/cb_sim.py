"""
cb_sim — Synthetic contextual-bandit environment and custom linear model.

Scenario
--------
A user has gone silent for N weeks (a fixed inactivity threshold).  The system
sends exactly ONE re-engagement push / email to try to bring them back.
The "context" describes the dormant user; the "action" is which message variant
to send (same offer, different tone/framing); the "reward" is whether the user
clicks through (1) or ignores it (0).

Features (context)
------------------
  lifetime_purchases — how many orders the user placed before going inactive
                       (Poisson-sampled, then standardized)
  days_as_customer   — how long they've had an account, in days
                       (min 30 days, Exp-sampled, then standardized)

Actions (K = 7 re-engagement message variants — same offer, different tone)
---------------------------------------------------------------------------
  1  "See what's new since you left"  — curiosity, highlights fresh content
  2  "Your personalized picks await"  — personalization, leverages their history
  3  "We miss you! Come back"         — emotional/warm, relationship-driven
  4  "Trending picks just for you"    — social proof, what others are buying
  5  "Flash deal: 24h only"           — urgency/scarcity, time-limited push
  6  "Your friends are shopping"      — peer pressure, leverages social graph
  7  "We saved your favorites"        — nostalgia, highlights past browsing

Ground-truth click model (unknown to the learner)
--------------------------------------------------
  p(click | x, a) = sigmoid(W_TRUE[a] · x)
  where x = [1, lifetime_purchases_std, days_as_customer_std]

  Intuition (which the bandit must *discover*):
    Two dominant segments:
      Loyal veterans (many purchases, long tenure)  → personalization works best
      Long-time browsers (few purchases, long tenure) → emotional appeal works best
    Two smaller niches:
      Burst buyers  (many purchases, short tenure)  → curiosity works best
      One-and-done  (few purchases, short tenure)   → social proof works best

  The last three variants (urgency, peer pressure, nostalgia) are never optimal
  — their biases are too low to compete anywhere in the feature space.
"""

import numpy as np
from collections import deque

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K = 7

ACTION_NAMES = {
    1: "See what's new since you left",
    2: "Your personalized picks await",
    3: "We miss you! Come back",
    4: "Trending picks just for you",
    5: "Flash deal: 24h only",
    6: "Your friends are shopping",
    7: "We saved your favorites",
}

FEATURE_NAMES = ["lifetime_purchases", "days_as_customer"]

FEATURE_STATS = {
    "lifetime_purchases": {"mean": 2.0, "std": 5.0},
    "days_as_customer":   {"mean": 90.0, "std": 180.0},
}

W_TRUE = np.array([
    [-3.0,  0.9,  -1.2],   # 1: curiosity — niche: burst buyers
    [-2.0,  0.6,   0.6],   # 2: personalization — dominant: loyal veterans
    [-2.0, -0.6,   0.6],   # 3: emotional — dominant: long-time browsers
    [-3.0, -0.9,  -1.2],   # 4: social proof — niche: one-and-done
    [-5.0,  1.0,  -0.5],   # 5: urgency — never competitive
    [-5.0, -0.3,   1.0],   # 6: peer pressure — never competitive
    [-4.0,  0.1,   0.1],   # 7: nostalgia — never competitive
])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class BanditEnv:
    """Synthetic CB environment: dormant user -> re-engagement message -> click/ignore.

    When noise_std > 0, each click outcome is perturbed by unobserved factors
    (mood, timing, device, etc.) that the features can't capture.  Concretely,
    we add Gaussian noise to a probit-corrected logit before sampling:

        p(click | x, a) = σ(κ · W_TRUE[a] · x  +  η),   η ~ N(0, noise_std²)
        κ = √(1 + π · noise_std² / 8)

    The correction factor κ ensures that E[click | x, a] = σ(W_TRUE[a] · x)
    regardless of noise level, counteracting the Jensen's-inequality bias that
    otherwise inflates average CTR (since most logits are negative).

    The features are always observed perfectly, but they don't fully explain
    click behaviour.  The model should still converge to W_TRUE, but it takes
    longer because each reward observation is noisier — making premature
    exploitation riskier.
    """

    def __init__(self, seed=0, noise_std=1.0, interaction_strength=0.0):
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.interaction_strength = interaction_strength

    def sample_context(self):
        """
        Sample a random dormant-user context.

        Returns
        -------
        x   : np.ndarray, shape (3,) — [1, purchases_std, customer_age_std]
        raw : dict with human-readable feature values
        """
        lifetime_purchases = int(max(0, self.rng.poisson(4)))
        days_as_customer = float(30.0 + self.rng.exponential(150))

        ps = FEATURE_STATS["lifetime_purchases"]
        ds = FEATURE_STATS["days_as_customer"]
        purchases_std = (lifetime_purchases - ps["mean"]) / ps["std"]
        days_std = (days_as_customer - ds["mean"]) / ds["std"]

        x = np.array([1.0, purchases_std, days_std])
        raw = {
            "lifetime_purchases": lifetime_purchases,
            "days_as_customer": round(days_as_customer),
        }
        return x, raw

    # Per-action interaction coefficients: how much x[1]*x[2] affects each action.
    _W_INTERACTION = np.array([0.5, -0.3, 0.8, -0.2, 0.4, -0.5, 0.1])

    def _base_logit(self, x, action):
        """True logit for (x, action), including any action-dependent interaction."""
        logit = float(W_TRUE[action - 1] @ x)
        if self.interaction_strength != 0:
            logit += self.interaction_strength * self._W_INTERACTION[action - 1] * x[1] * x[2]
        return logit

    def click_prob(self, x, action):
        """Expected P(click | x, action) — the noiseless true probability."""
        return float(sigmoid(self._base_logit(x, action)))

    def sample_click(self, x, action):
        """Sample a Bernoulli click, with optional unobserved-factor noise.

        When noise_std > 0, we scale the logit by κ = √(1 + πσ²/8) before
        adding noise so that E[click | x, a] = σ(W_TRUE[a]·x) regardless of
        noise level.  Without this correction, noise inflates average CTR via
        Jensen's inequality (most logits are negative → convex region of σ).
        """
        logit = self._base_logit(x, action)
        if self.noise_std > 0:
            kappa = np.sqrt(1.0 + np.pi * self.noise_std ** 2 / 8.0)
            p = sigmoid(kappa * logit + self.rng.normal(0, self.noise_std))
        else:
            p = sigmoid(logit)
        return 1 if self.rng.random() < p else 0

    def oracle_action(self, x):
        """Return the action with the highest expected CTR for this context."""
        ctrs = [self.click_prob(x, a) for a in range(1, K + 1)]
        return int(np.argmax(ctrs)) + 1

    def oracle_ctr(self, N=50_000, seed=999):
        """Monte-Carlo oracle CTR (picks best action, but still subject to noise)."""
        env_mc = BanditEnv(seed=seed, noise_std=self.noise_std,
                          interaction_strength=self.interaction_strength)
        total = 0
        for _ in range(N):
            x, _ = env_mc.sample_context()
            best = env_mc.oracle_action(x)
            total += env_mc.sample_click(x, best)
        return total / N


# ---------------------------------------------------------------------------
# Custom linear bandit model
# ---------------------------------------------------------------------------
class LinearBanditModel:
    """
    Per-action online logistic regression for contextual bandits.

    Each action a ∈ {1..K} has its own weight vector w_a of shape (d,).
    Prediction: p_hat(click | x, a) = sigmoid(w_a · x)
    Policy: epsilon-greedy over predicted click probabilities.
    """

    def __init__(self, n_actions=K, n_features=3, lr=0.1):
        self.K = n_actions
        self.d = n_features
        self.lr = lr
        self.W = np.zeros((n_actions, n_features))

    def predict_proba(self, x):
        """Predicted P(click) for each action. Returns shape (K,)."""
        return sigmoid(self.W @ x)

    def action_probs(self, x, epsilon):
        """Epsilon-greedy distribution over actions. Returns shape (K,)."""
        logits = self.W @ x
        best = np.argmax(logits)
        probs = np.full(self.K, epsilon / self.K)
        probs[best] += 1.0 - epsilon
        return probs

    def sample_action(self, x, epsilon, rng):
        """Sample from epsilon-greedy. Returns (1-indexed action, propensity)."""
        probs = self.action_probs(x, epsilon)
        idx = rng.choice(self.K, p=probs)
        return int(idx + 1), float(probs[idx])

    def update(self, x, action, reward, propensity, method="direct",
               reward_model=None):
        """
        Update model weights after observing (x, action, reward).

        Methods
        -------
        direct : SGD on the chosen action only. Simple and low-variance, but the
                 effective learning rate differs across actions when exploration
                 is non-uniform (exploited actions get more updates).

        ips    : Weight the gradient by 1/propensity. Gives each action the same
                 expected gradient magnitude regardless of how often it's chosen.
                 Unbiased but higher variance, especially when propensity is small.

        dr     : Doubly Robust. Requires a separate `reward_model` that predicts
                 expected rewards for all actions. For each action k, constructs a
                 DR pseudo-reward:
                   chosen:   r_dr = r_hat(k) + (r_observed - r_hat(k)) / propensity
                   unchosen: r_dr = r_hat(k)
                 Then updates the policy model toward these targets.
                 "Doubly robust" because the estimate is consistent if *either*
                 the reward model or the propensities are correct.
        """
        a = action - 1

        if method == "direct":
            p_hat = float(sigmoid(self.W[a] @ x))
            grad = (p_hat - reward) * x
            self.W[a] -= self.lr * grad

        elif method == "ips":
            p_hat = float(sigmoid(self.W[a] @ x))
            grad = (p_hat - reward) * x
            self.W[a] -= self.lr * grad / propensity

        elif method == "dr":
            if reward_model is None:
                raise ValueError("DR requires a separate reward_model")
            p_hat_all = self.predict_proba(x)
            r_hat_all = reward_model.predict_proba(x)

            for k in range(self.K):
                if k == a:
                    r_dr = r_hat_all[k] + (reward - r_hat_all[k]) / propensity
                else:
                    r_dr = r_hat_all[k]
                r_dr = np.clip(r_dr, -2.0, 3.0)
                grad = (p_hat_all[k] - r_dr) * x
                self.W[k] -= self.lr * grad

        else:
            raise ValueError(f"Unknown method: {method}")

    def copy(self):
        """Return an independent copy of this model."""
        m = LinearBanditModel(self.K, self.d, self.lr)
        m.W = self.W.copy()
        return m


# ---------------------------------------------------------------------------
# Bandit loop (custom model)
# ---------------------------------------------------------------------------
def run_bandit(model, env, N=5000, epsilon=0.1, method="direct",
               report_every=500, log=False):
    """
    Run N rounds of the contextual-bandit loop with a LinearBanditModel.

    When method='dr', a separate reward model is created and trained alongside
    the policy model.  The reward model learns to predict rewards via direct SGD
    on every observed (x, action, reward) tuple.  Its predictions are then used
    as the baseline in the DR update for the policy model — providing
    "hallucinated" rewards for unchosen actions and an IPS-corrected target for
    the chosen action.

    Returns
    -------
    ctr : float — overall click-through rate
    logs : dict (only if log=True) — arrays of xs, actions, props, clicks
           for downstream OPE
    """
    clicks = 0
    recent = deque(maxlen=report_every)

    reward_model = None
    if method == "dr":
        reward_model = LinearBanditModel(model.K, model.d, model.lr)

    if log:
        xs_log = np.zeros((N, model.d - 1), dtype=np.float32)
        a_log = np.zeros(N, dtype=np.int32)
        p_log = np.zeros(N, dtype=np.float32)
        c_log = np.zeros(N, dtype=np.int8)

    for t in range(1, N + 1):
        x, _ = env.sample_context()

        action, prop = model.sample_action(x, epsilon, env.rng)
        click = env.sample_click(x, action)

        if method == "dr":
            model.update(x, action, click, prop, method="dr",
                         reward_model=reward_model)
            reward_model.update(x, action, click, prop, method="direct")
        else:
            model.update(x, action, click, prop, method=method)

        clicks += click
        recent.append(click)

        if log:
            xs_log[t - 1] = x[1:]
            a_log[t - 1] = action
            p_log[t - 1] = prop
            c_log[t - 1] = click

        if report_every and t % report_every == 0:
            probs = model.action_probs(x, epsilon)
            print(
                f"t={t:5d}  overall_CTR={clicks/t:.3f}  "
                f"recent_CTR={np.mean(recent):.3f}  "
                f"probs={np.round(probs, 3)}"
            )

    ctr = clicks / N
    if log:
        logs = {"xs": xs_log, "actions": a_log, "props": p_log, "clicks": c_log}
        return ctr, logs
    return ctr


# ---------------------------------------------------------------------------
# Off-policy evaluation
# ---------------------------------------------------------------------------
def ope_ips_snips(model, logs, epsilon, mu=None):
    """
    Estimate the CTR that `model` with the given `epsilon` would achieve,
    using logged data collected under a (possibly non-uniform) logging policy.

    Parameters
    ----------
    mu : float or None
        If float, use as a constant propensity for all observations (valid when
        the logging policy was uniform, e.g. warm-up with ε=1 → mu=0.25).
        If None, use the per-instance propensities stored in logs['props'].

    Returns (ips_estimate, snips_estimate).
    """
    xs, actions, clicks = logs["xs"], logs["actions"], logs["clicks"]
    logged_props = logs["props"]
    N = len(actions)
    weights = np.empty(N, dtype=np.float64)

    for i in range(N):
        x = np.array([1.0, xs[i, 0], xs[i, 1]])
        probs = model.action_probs(x, epsilon)
        pi_ai = probs[int(actions[i]) - 1]
        mu_i = mu if mu is not None else float(logged_props[i])
        weights[i] = pi_ai / mu_i

    ips = float(np.mean(weights * clicks))
    snips = float(np.sum(weights * clicks) / np.sum(weights))
    return ips, snips
