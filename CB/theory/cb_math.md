## Scratch Notes

> *Assumes online learning with Doubly Robust cost estimation (`--cb_type dr` in VW).  DR is what allows the model to update weights for all $K$ actions on every round — not just the one that was chosen.  See Section 5 for details.*

Step-by-step walkthrough of what happens, in order:

**Step 1 — Warmup: collect unbiased data.**  We serve actions uniformly at random, so $\mu(a \mid x) = 1/K$ for every action. This is a valid probability distribution: $\sum_{a=1}^{K} \mu(a \mid x) = 1$.  Every action gets equal airtime regardless of context.

**Step 2 — Train the model (during warmup).**  We learn $K$ independent linear models — one per action — each with its own weight vector $\hat{W}[a]$.  For a given context $x$, each model produces a score $\hat{r}(x, a) = \hat{W}[a] \cdot x$.  This is a raw linear output, **not** a probability — it can land outside $[0,1]$ and is not passed through a sigmoid.  The output is a raw linear score — not a calibrated probability — but it preserves the **ranking** of actions for a given context, which is all that matters: the $\varepsilon$-greedy policy uses $\arg\max_a \hat{r}(x, a)$ to decide which action to favor.

**How the model learns: DR cost targets + squared loss.**  The model needs a cost target for every action on every round to update all $K$ weight vectors — but we only observe the outcome for the one action we chose.  Doubly Robust fills in the gaps.

First, notation — VW works in **costs** (lower is better), not rewards:

- $a_{\text{chosen}}$ — the action we actually selected and served on this round
- $a'$ — a generic action index ranging over all $a' \in \{1, \dots, K\}$; we need a cost estimate for each
- $c_{\text{observed}} \in \{0, 1\}$ — the observed cost for $a_{\text{chosen}}$: click $\to 0$ (good), no click $\to 1$ (bad)
- $\hat{c}_{\text{model}}(a') = \hat{W}[a'] \cdot x$ — the model's current predicted cost for action $a'$

For the **chosen action** ($a' = a_{\text{chosen}}$), we have the actual outcome, so DR blends it with the model's own prediction:

$$\hat{c}_{\text{DR}}(a') = \hat{c}_{\text{model}}(a') + \frac{c_{\text{observed}} - \hat{c}_{\text{model}}(a')}{\mu(a \mid x)}$$

Read this as: start with the model's prediction, then add a correction for how wrong it was — scaled by $1/\mu$ to account for how likely we were to have seen this action at all.  If the model was already right ($c_{\text{observed}} \approx \hat{c}_{\text{model}}$), the correction is tiny.  If it was wrong, the correction is large.

For **unchosen actions** ($a' \neq a_{\text{chosen}}$), we have no observed outcome, so we fall back to the model's current prediction:

$$\hat{c}_{\text{DR}}(a') = \hat{c}_{\text{model}}(a')$$

This is the best we can do — the model teaches itself about actions it didn't try, using whatever it's learned so far.

With cost targets for all $K$ actions in hand, each action's weights are updated via SGD on **squared loss**:

$$\hat{W}[a'] \leftarrow \hat{W}[a'] - \eta \cdot \bigl(\hat{W}[a'] \cdot x - \hat{c}_{\text{DR}}(a')\bigr) \cdot x$$

where $\eta$ is the learning rate.  The loss function is squared error, not cross-entropy — this is regression, not classification.

**Why squared loss → monotonic relationship with click probability.**  With squared loss on binary cost targets ($c = 0$ for click, $c = 1$ for no click), the optimal prediction for a given $(x, a)$ converges to the conditional expected cost:

$$\hat{W}[a] \cdot x \;\to\; \mathbb{E}[c \mid x, a] = 1 - p(\text{click} \mid x, a)$$

So the model's output tracks $1 - p(\text{click})$: lower predicted cost = higher click probability.  The output isn't a probability (no sigmoid, can land outside $[0,1]$), but it's **monotonically related** to the true click probability — and monotonicity is all the $\arg\max$ needs.  If the model correctly ranks $\hat{W}[a] \cdot x < \hat{W}[b] \cdot x$ (action $a$ has lower predicted cost), then action $a$ has higher click probability, and the $\varepsilon$-greedy formula will favor it.

**Step 3 — Build a candidate policy.**  Given the trained scores $\hat{r}(x, a)$, the $\varepsilon$-greedy formula converts them into a probability distribution $\pi(\cdot \mid x)$ over the $K$ actions.  The greedy best action (highest $\hat{r}$) gets probability $(1 - \varepsilon + \varepsilon/K)$; every other action gets $\varepsilon/K$.  This sums to 1:

$$\underbrace{(1 - \varepsilon + \varepsilon/K)}_{\text{best action}} + \underbrace{(K-1) \cdot \varepsilon/K}_{\text{all others}} = 1 - \varepsilon + \varepsilon/K + \varepsilon - \varepsilon/K = 1$$

For $\varepsilon = 0.05$, $K = 4$: best action gets $0.9625$, each other gets $0.0125$, total $= 1$.

**Step 4 — Compute importance weights.**  For each logged observation $i$, we compare how much $\pi$ favors the action that was taken vs. how much $\mu$ did: $w_i = \pi(a_i \mid x_i) / \mu(a_i \mid x_i)$.  Computing $\pi(a_i \mid x_i)$ is deterministic — just evaluate the $\varepsilon$-greedy formula for the logged context and action.  No sampling needed.

**Step 5 — SNIPS picks the best $\varepsilon$.**  For each candidate $\varepsilon$, repeat steps 3–4 and compute $\hat{V}_{\text{SNIPS}}$.  The reweighting effectively reconstructs what $\pi$'s data would have looked like: observations of actions that $\pi$ favors get amplified, and observations of actions $\pi$ avoids get suppressed.  Since the environment's rewards don't depend on which policy chose the action, a dataset with $\pi$'s action frequencies gives $\pi$'s expected reward.  Pick the $\varepsilon$ with the highest SNIPS score.

**Step 6 — Deploy and keep learning.**  Load the model with the best $\varepsilon$ and run live.  Now the model's scores directly control which action gets served, and each new observation updates $\hat{W}$ — creating a feedback loop that improves over time.