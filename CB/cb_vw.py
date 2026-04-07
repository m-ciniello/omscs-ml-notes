"""
cb_vw — VowpalWabbit-specific helpers for contextual-bandit experiments.

These are kept separate from cb_sim so the core environment and custom models
don't depend on the vowpalwabbit package.
"""

import numpy as np
from collections import deque
from vowpalwabbit import PredictionType

from cb_sim import BanditEnv, K


def vw_features(x):
    """Format standardized context vector as a VW feature string."""
    return f"| lifetime_purchases:{x[1]:.4f} days_as_customer:{x[2]:.4f}"


def sample_from_probs(probs, rng):
    """
    Sample one action from a VW action-probability distribution.

    Returns (action in 1..K, propensity of that action).
    """
    a_idx = rng.choice(len(probs), p=np.array(probs))
    return int(a_idx + 1), float(probs[a_idx])


def run_bandit(vw, env, N=5000, report_every=500):
    """
    Run N rounds of: predict -> sample action -> observe reward -> learn.

    Returns overall CTR.
    """
    clicks = 0
    recent = deque(maxlen=report_every)

    for t in range(1, N + 1):
        x, _ = env.sample_context()
        feats = vw_features(x)

        probs = vw.predict(feats, prediction_type=PredictionType.ACTION_PROBS)
        action, prop = sample_from_probs(probs, env.rng)

        click = env.sample_click(x, action)
        cost = 1 - click
        vw.learn(f"{action}:{cost}:{prop} {feats}")

        clicks += click
        recent.append(click)

        if t % report_every == 0:
            print(
                f"t={t:5d}  overall_CTR={clicks/t:.3f}  "
                f"recent_CTR={np.mean(recent):.3f}  "
                f"probs={np.round(probs, 3)}"
            )

    return clicks / N
