# Hidden Markov Models
### A Step-by-Step Derivation of the Three Classic Algorithms
*OMSCS ML | Probabilistic Models Unit*

---

## Background: What Problem Do HMMs Solve?

Many real-world processes generate sequences of observations over time — a person speaking a sentence, a gene expressed across positions in a genome, a financial instrument ticking through prices. In all of these cases:

1. There is some **hidden state** driving the process (the phoneme being spoken, the functional region of the genome, the market regime)
2. We can only **observe noisy emissions** from that state (the audio waveform, the nucleotide, the price)
3. The hidden state **evolves over time** according to some probabilistic rule

An HMM is the simplest probabilistic model that captures all three of these properties simultaneously. Before defining it formally, it helps to understand what we need the model to *do*:

- **Evaluation**: Given a trained model and a sequence of observations, how likely is that sequence? (e.g., does this audio clip match the word "hello"?)
- **Decoding**: Given a trained model and a sequence of observations, what is the most likely sequence of hidden states that produced it? (e.g., what phonemes were spoken?)
- **Learning**: Given only observations (no hidden states), how do we fit the model parameters? (e.g., train a speech recognizer from audio transcripts)

Each of these tasks has a dedicated algorithm. The rest of these notes derive all three from scratch.

---

## Part 1: The Model

### Formal Definition

An HMM is defined by five components:

$$\lambda = (N, M, \mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$$

- $N$ — the number of hidden states, indexed $\{1, 2, \ldots, N\}$
- $M$ — the number of distinct observation symbols, indexed $\{1, 2, \ldots, M\}$
- $\mathbf{A} \in \mathbb{R}^{N \times N}$ — the **transition matrix**, where $A_{ij} = P(q_{t+1} = j \mid q_t = i)$ is the probability of transitioning from state $i$ to state $j$
- $\mathbf{B} \in \mathbb{R}^{N \times M}$ — the **emission matrix**, where $B_{ik} = P(o_t = k \mid q_t = i)$ is the probability of emitting observation $k$ from state $i$
- $\boldsymbol{\pi} \in \mathbb{R}^N$ — the **initial state distribution**, where $\pi_i = P(q_1 = i)$

Each row of $\mathbf{A}$ sums to 1 (it is a valid probability distribution over next states). Each row of $\mathbf{B}$ sums to 1 (it is a valid probability distribution over observations). $\boldsymbol{\pi}$ sums to 1.

### The Two Core Independence Assumptions

HMMs are tractable because of two assumptions that dramatically simplify the joint distribution:

**Assumption 1 — The Markov Property**: The current hidden state depends only on the immediately preceding hidden state, not on any earlier history:

$$P(q_t \mid q_{t-1}, q_{t-2}, \ldots, q_1) = P(q_t \mid q_{t-1}) = A_{q_{t-1}, q_t}$$

This is the "hidden Markov" part of the name. It means the state sequence $q_1, q_2, \ldots, q_T$ is a **Markov chain** — a memoryless random process where the future depends only on the present, not the past.

**Assumption 2 — The Output Independence Assumption**: Each observation depends only on the hidden state that generated it, not on any other states or observations:

$$P(o_t \mid q_1, \ldots, q_T, o_1, \ldots, o_T) = P(o_t \mid q_t) = B_{q_t, o_t}$$

### The Joint Distribution

Given these two assumptions, we can derive the joint probability of a state sequence $\mathbf{q} = q_1, \ldots, q_T$ and observation sequence $\mathbf{o} = o_1, \ldots, o_T$ from first principles.

Start with the chain rule of probability, factoring observations and states in time order:

$$P(\mathbf{o}, \mathbf{q} \mid \lambda) = P(q_1) \cdot P(o_1 \mid q_1) \cdot P(q_2 \mid q_1, o_1) \cdot P(o_2 \mid q_2, q_1, o_1) \cdots$$

Now apply the two independence assumptions to simplify each factor:

- $P(q_1) = \pi_{q_1}$ — by definition of the initial distribution
- $P(o_1 \mid q_1) = B_{q_1, o_1}$ — by the output independence assumption
- $P(q_t \mid q_{t-1}, \ldots, q_1, o_{t-1}, \ldots, o_1) = P(q_t \mid q_{t-1}) = A_{q_{t-1}, q_t}$ — by the Markov property, the state depends only on the previous state
- $P(o_t \mid q_t, q_{t-1}, \ldots, o_{t-1}, \ldots) = P(o_t \mid q_t) = B_{q_t, o_t}$ — by output independence, the observation depends only on the current state

Substituting all of these simplifications, the joint factorizes cleanly as:

$$\boxed{P(\mathbf{o}, \mathbf{q} \mid \lambda) = \pi_{q_1} \cdot B_{q_1, o_1} \cdot \prod_{t=2}^{T} A_{q_{t-1}, q_t} \cdot B_{q_t, o_t}}$$

Reading left to right: start in state $q_1$ with probability $\pi_{q_1}$, emit $o_1$ with probability $B_{q_1, o_1}$, then for each subsequent step transition to $q_t$ with probability $A_{q_{t-1}, q_t}$ and emit $o_t$ with probability $B_{q_t, o_t}$. The two independence assumptions are what allow each factor to collapse from a high-dimensional conditional down to a single matrix entry.

### Model Topology: Ergodic vs. Left-Right HMMs

So far we have placed no constraints on $\mathbf{A}$ beyond each row summing to 1. This gives an **ergodic** (fully connected) model where any state can transition to any other state in a single step — $A_{ij} > 0$ for all $i, j$. But in many real-world problems, the underlying process has inherent directionality: time moves forward, and states should too.

A **left-right** (or Bakis) model enforces this by constraining:

$$A_{ij} = 0 \quad \text{for all } j < i$$

States can only transition to themselves (self-loop) or to states with a higher index — never backward. The initial state distribution is also constrained to $\pi_1 = 1$, $\pi_i = 0$ for $i \neq 1$, since the sequence always begins at state 1. The transition matrix therefore has an upper-triangular structure:

$$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & a_{13} & \cdots & a_{1N} \\ 0 & a_{22} & a_{23} & \cdots & a_{2N} \\ 0 & 0 & a_{33} & \cdots & a_{3N} \\ \vdots & & & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & a_{NN} \end{pmatrix}$$

An additional constraint often used in practice is a **bounded jump**: $A_{ij} = 0$ for $j > i + \Delta$ for some small $\Delta$ (e.g., $\Delta = 2$). This prevents the model from skipping too many states at once, which would be unphysical for a smooth temporal process like speech.

**Why topology matters — and its connection to modern ML:**

Left-right models are the standard for any process where order is irreversible: speech phonemes, video frames, genomic regions. Ergodic models are appropriate when states recur freely, such as weather patterns or market regimes.

This topological constraint is the HMM analog of **architectural inductive biases** in deep learning. Just as a convolutional network encodes translational invariance by sharing weights across positions, a left-right HMM encodes temporal directionality by zeroing out backward transitions. The transformer's causal masking (setting attention weights to $-\infty$ for future positions) is the direct neural analog: it enforces that position $t$ cannot attend to positions $> t$, exactly as a left-right HMM enforces that state $i$ cannot transition to state $j < i$.

Crucially, zeroed-out entries in $\mathbf{A}$ remain zero throughout Baum-Welch — a parameter initialized to zero receives zero gradient from the soft counts, so the topology is preserved across all training iterations. This is a clean example of **hard constraints vs. soft regularization**: topology is enforced structurally, not encouraged probabilistically.

---

## Part 2: The Evaluation Problem — The Forward Algorithm

### What We Want

Given a model $\lambda$ and observation sequence $\mathbf{o} = o_1, \ldots, o_T$, compute:

$$P(\mathbf{o} \mid \lambda) = \sum_{\text{all state sequences } \mathbf{q}} P(\mathbf{o}, \mathbf{q} \mid \lambda)$$

We marginalize out the hidden states — we don't care which state sequence was taken, only the total probability of the observations.

### Why Naïve Summation Fails

The most direct approach: enumerate all possible state sequences, compute the joint probability of each, and sum. There are $N^T$ possible state sequences of length $T$. For $N = 5$ states and $T = 100$ time steps, that is $5^{100} \approx 10^{69}$ terms. This is completely intractable.

The forward algorithm reduces this to $O(N^2 T)$ using **dynamic programming** — the key insight being that we can reuse partial computations rather than recomputing from scratch for every state sequence.

### Defining the Forward Variable

Define the **forward variable** $\alpha_t(i)$ as the probability of two things simultaneously:

1. Having observed the partial sequence $o_1, o_2, \ldots, o_t$ so far
2. Being in state $i$ at time $t$

Formally:

$$\alpha_t(i) = P(o_1, o_2, \ldots, o_t, q_t = i \mid \lambda)$$

### Initialization

At $t = 1$, we haven't yet used any transition probabilities — we just start in some state and emit the first observation:

$$\alpha_1(i) = \pi_i \cdot B_{i, o_1} \quad \text{for all } i \in \{1, \ldots, N\}$$

This is the probability of starting in state $i$ times the probability of emitting $o_1$ from state $i$.

### Recursion

Now suppose we know $\alpha_t(i)$ for all states $i$. How do we compute $\alpha_{t+1}(j)$?

By definition:

$$\alpha_{t+1}(j) = P(o_1, \ldots, o_{t+1}, q_{t+1} = j \mid \lambda)$$

We condition on what state we were in at time $t$, using the law of total probability:

$$\alpha_{t+1}(j) = \sum_{i=1}^{N} P(o_1, \ldots, o_{t+1}, q_t = i, q_{t+1} = j \mid \lambda)$$

Now we factor this joint probability step by step. First apply the chain rule, peeling off the transition and then the emission:

$$P(o_1, \ldots, o_{t+1}, q_t = i, q_{t+1} = j \mid \lambda)$$
$$= P(o_1, \ldots, o_t, q_t = i \mid \lambda) \cdot P(q_{t+1} = j \mid q_t = i, o_1, \ldots, o_t) \cdot P(o_{t+1} \mid q_{t+1} = j, q_t = i, o_1, \ldots, o_t)$$

Now apply the independence assumptions to simplify the last two factors:

- $P(q_{t+1} = j \mid q_t = i, o_1, \ldots, o_t) = P(q_{t+1} = j \mid q_t = i) = A_{ij}$ — by the Markov property, the next state depends only on the current state, not on the history of observations
- $P(o_{t+1} \mid q_{t+1} = j, q_t = i, o_1, \ldots, o_t) = P(o_{t+1} \mid q_{t+1} = j) = B_{j, o_{t+1}}$ — by output independence, the observation depends only on the current state

Substituting back and recognizing $P(o_1, \ldots, o_t, q_t = i \mid \lambda) = \alpha_t(i)$:

$$= \sum_{i=1}^{N} \underbrace{P(o_1, \ldots, o_t, q_t = i \mid \lambda)}_{\alpha_t(i)} \cdot \underbrace{P(q_{t+1} = j \mid q_t = i)}_{A_{ij}} \cdot \underbrace{P(o_{t+1} \mid q_{t+1} = j)}_{B_{j, o_{t+1}}}$$

The three terms correspond to: (1) being in state $i$ after seeing $o_1, \ldots, o_t$, (2) transitioning to state $j$, (3) emitting $o_{t+1}$ from state $j$. This gives us the recursion:

$$\boxed{\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \underbrace{\alpha_t(i)}_{\text{prob. of reaching state } i \text{ at } t} \cdot \underbrace{A_{ij}}_{\text{transition } i \to j}\right] \cdot \underbrace{B_{j, o_{t+1}}}_{\text{emit } o_{t+1} \text{ from } j}}$$

### Termination

Once we have $\alpha_T(i)$ for all states, the total observation probability is just the sum over all final states:

$$P(\mathbf{o} \mid \lambda) = \sum_{i=1}^{N} \alpha_T(i)$$

To see why, recall what $\alpha_T(i)$ actually is:

$$\alpha_T(i) = P(o_1, \ldots, o_T, q_T = i \mid \lambda)$$

This is the **joint** probability of the entire observation sequence *and* ending in state $i$ — not a conditional. We want $P(\mathbf{o} \mid \lambda)$, the probability of the observation sequence regardless of which state we ended in. Since the HMM must be in exactly one state at time $T$, the events $\{q_T = 1\}, \{q_T = 2\}, \ldots, \{q_T = N\}$ are mutually exclusive and exhaustive. By the law of total probability we can therefore marginalize out the final state:

$$P(\mathbf{o} \mid \lambda) = \sum_{i=1}^{N} P(\mathbf{o}, q_T = i \mid \lambda) = \sum_{i=1}^{N} \alpha_T(i)$$

Concretely: each $\alpha_T(i)$ accounts for every possible path through the trellis that ends in state $i$ and generates the full observation sequence. Summing over all $i$ collects every possible path regardless of where it ends — which is exactly $P(\mathbf{o} \mid \lambda)$. It's the same logic as computing the probability of rain by summing $P(\text{rain, umbrella})$ and $P(\text{rain, no umbrella})$ — the umbrella outcome is irrelevant, so you sum it out.

### Complexity

- Initialization: $O(N)$
- Recursion: $T-1$ steps, each requiring $O(N^2)$ work (for each of $N$ destination states, we sum over $N$ source states)
- Termination: $O(N)$
- **Total: $O(N^2 T)$** — a dramatic improvement over $O(N^T)$

---

## Part 3: The Decoding Problem — The Viterbi Algorithm

### What We Want

Rather than summing over all state sequences, we now want the **single most likely** state sequence given the observations:

$$\mathbf{q}^* = \arg\max_{\mathbf{q}} P(\mathbf{q} \mid \mathbf{o}, \lambda) = \arg\max_{\mathbf{q}} P(\mathbf{o}, \mathbf{q} \mid \lambda)$$

(The second equality holds because $P(\mathbf{o} \mid \lambda)$ is the same for all state sequences and doesn't affect the argmax.)

**A note on $\arg\max$ vs $\max$:** these are easy to conflate but mean different things. $\max$ returns the *value* — the actual probability number. $\arg\max$ returns the *input that achieved that value* — here, the state sequence itself. If three state sequences have probabilities 0.1, 0.7, and 0.2, then $\max = 0.7$ and $\arg\max = $ sequence 2. In Viterbi, we don't care what the best path's probability *is* — we care about *which path* it is, so the goal is $\arg\max$. The $\max$ (without arg) does appear at termination when we write $P^* = \max_i \, \delta_T(i)$ — there we're reading off the score in order to identify which final state to start backtracking from.

### The Viterbi Variable

The structure mirrors the forward algorithm exactly, except we replace the **sum** over previous states with a **max**. Define:

$$\delta_t(i) = \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_{t-1}, q_t = i, o_1, \ldots, o_t \mid \lambda)$$

This is the probability of the **most probable path** — under the model $\lambda$ — that ends in state $i$ at time $t$ and jointly generates observations $o_1, \ldots, o_t$. The maximization is over all possible choices of the preceding states $q_1, \ldots, q_{t-1}$; we are asking: given that we must be in state $i$ at time $t$, what is the highest-probability way to have gotten here?

We also define a **backpointer** $\psi_t(j)$ that records which state $i$ at time $t-1$ led to the best path ending in state $j$ at time $t$. This lets us recover the full state sequence at the end.

### Initialization

$$\delta_1(i) = \pi_i \cdot B_{i, o_1}$$

$$\psi_1(i) = 0 \quad \text{(no predecessor at } t=1\text{)}$$

### Recursion

The recursion is identical to the forward algorithm except $\sum$ becomes $\max$:

$$\delta_{t+1}(j) = \left[\max_{i} \, \underbrace{\delta_t(i)}_{\text{best prob. of reaching state } i \text{ at } t} \cdot \underbrace{A_{ij}}_{\text{transition } i \to j}\right] \cdot \underbrace{B_{j, o_{t+1}}}_{\text{emit } o_{t+1} \text{ from } j}$$

Reading left to right: $\delta_t(i)$ is the probability of the most probable path that ends in state $i$ at time $t$ — our best score so far. Multiplying by $A_{ij}$ extends that path by one step, transitioning to state $j$. We take the $\max$ over all possible predecessor states $i$ because we only want to carry forward the single best way of arriving at $j$ — all worse predecessors are discarded. Finally, multiplying by $B_{j, o_{t+1}}$ accounts for emitting the next observation from state $j$.

And we record which $i$ achieved that maximum:

$$\psi_{t+1}(j) = \arg\max_{i} \, \delta_t(i) \cdot A_{ij}$$

### Why max and sum are interchangeable here

In the forward algorithm, we summed because we wanted total probability — contributions from all paths. Here, we want the single best path, so we only keep the maximum-probability predecessor at each step. The dynamic programming principle holds in both cases: the best path to state $j$ at time $t+1$ must pass through whichever state $i$ at time $t$ maximizes the product of the best path to $i$ and the transition $A_{ij}$.

### Termination and Backtracking

The probability of the best overall path is:

$$P^* = \max_i \, \delta_T(i)$$

The final state of the best path is:

$$q_T^* = \arg\max_i \, \delta_T(i)$$

We then **backtrack** through the $\psi$ pointers to recover the full sequence:

$$q_t^* = \psi_{t+1}(q_{t+1}^*) \quad \text{for } t = T-1, T-2, \ldots, 1$$

### Complexity

Same as the forward algorithm: $O(N^2 T)$.

### The Log-Domain Trick

In practice, products of many probabilities underflow to zero in floating point. The standard fix is to work in log space. Since $\log$ is monotonically increasing, it preserves the argmax:

$$\log \delta_{t+1}(j) = \max_i \left[\log \delta_t(i) + \log A_{ij}\right] + \log B_{j, o_{t+1}}$$

Products become sums, and underflow is eliminated. We will revisit the scaling problem more carefully after introducing Baum-Welch.

### Posterior Decoding: An Alternative to Viterbi

Viterbi finds the **jointly** most likely state sequence — the single path $\mathbf{q}^*$ that maximizes $P(\mathbf{q} \mid \mathbf{o}, \lambda)$ as a whole. But this is not the only sensible answer to the decoding question. A different objective is to maximize **per-position accuracy**: for each time step $t$ independently, choose the state that is most likely given all the observations.

This is called **posterior decoding**, and it uses the $\gamma$ variable (derived in Part 4 below) rather than $\delta$:

$$\hat{q}_t = \arg\max_i \, \gamma_t(i) = \arg\max_i \, P(q_t = i \mid \mathbf{o}, \lambda)$$

At each position $t$, we pick the single most probable state marginally — that is, after summing over all possible configurations of every other time step.

**When do Viterbi and posterior decoding disagree?**

They can give genuinely different answers, and neither is universally better. Consider a simple example: suppose at time $t$, the most probable marginal state is $i$, but state $i$ at time $t$ cannot be followed by any high-probability state at time $t+1$ given $o_{t+2}, \ldots$. Viterbi will avoid this path because it sees the full sequence; posterior decoding will pick state $i$ at time $t$ anyway because it only looks at the marginal. Concretely:

- **Viterbi is better** when you need a *globally coherent* sequence — e.g., enforcing grammatical constraints in part-of-speech tagging, where $\hat{q}_t$ and $\hat{q}_{t+1}$ must be jointly valid.
- **Posterior decoding is better** when per-position accuracy is what matters — e.g., gene finding, where you care about correctly labeling each nucleotide and a low-probability transition between two regions shouldn't force you to mislabel either.

A subtle point: the sequence $(\hat{q}_1, \hat{q}_2, \ldots, \hat{q}_T)$ produced by posterior decoding is **not guaranteed to be a valid path** under the HMM — it may include transitions with zero probability (if $A_{ij} = 0$ for some consecutive $(\hat{q}_t, \hat{q}_{t+1})$ pair). Viterbi always produces a valid path by construction.

---

## Part 4: The Learning Problem — The Baum-Welch Algorithm

### What We Want

Given only observations $\mathbf{o} = o_1, \ldots, o_T$ (no hidden state labels), find parameters $\lambda = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$ that maximize the likelihood:

$$\lambda^* = \arg\max_\lambda P(\mathbf{o} \mid \lambda)$$

This is an **unsupervised learning** problem — we must infer both the parameters and implicitly the hidden states at the same time. Direct maximization is intractable because the hidden states are unobserved. Baum-Welch is an application of the **Expectation-Maximization (EM) algorithm** to this problem — we will derive it from first principles below.

### The EM Framework (Brief)

EM applies whenever we have latent (hidden) variables. The key idea is to alternate between two steps:

- **E-step**: Compute the expected value of the hidden variables given the current parameters. This "fills in" the missing data in a soft, probabilistic way.
- **M-step**: Re-estimate the parameters as if the expected hidden variables were the true observed data.

Each iteration is guaranteed to not decrease the log-likelihood, and under mild conditions converges to a local maximum. We will now derive the E and M steps concretely for the HMM.

### The Backward Variable

The forward variable $\alpha_t(i)$ captures probability looking forward from the start. To perform the E-step we also need a **backward variable** $\beta_t(i)$ that captures probability looking backward from the end:

$$\beta_t(i) = P(o_{t+1}, o_{t+2}, \ldots, o_T \mid q_t = i, \lambda)$$

This is the probability of seeing the rest of the observation sequence $o_{t+1}, \ldots, o_T$, given that we are in state $i$ at time $t$.

**Initialization** (at the end of the sequence):

$$\beta_T(i) = 1 \quad \text{for all } i$$

This says: given we are at time $T$, the probability of observing the empty remaining sequence is 1 (there is nothing left to observe).

**Recursion** (running backward from $T-1$ to $1$):

$$\beta_t(i) = \sum_{j=1}^{N} A_{ij} \cdot B_{j, o_{t+1}} \cdot \beta_{t+1}(j)$$

To derive this, start from the definition and condition on the next state $j$ using the law of total probability:

$$\beta_t(i) = P(o_{t+1}, \ldots, o_T \mid q_t = i, \lambda) = \sum_{j=1}^{N} P(o_{t+1}, \ldots, o_T, q_{t+1} = j \mid q_t = i, \lambda)$$

Now apply the chain rule to factor the joint inside the sum, peeling off the transition, then the emission at $t+1$, then the rest:

$$P(o_{t+1}, \ldots, o_T, q_{t+1} = j \mid q_t = i, \lambda)$$
$$= \underbrace{P(q_{t+1} = j \mid q_t = i)}_{A_{ij}} \cdot \underbrace{P(o_{t+1} \mid q_{t+1} = j)}_{B_{j, o_{t+1}}} \cdot \underbrace{P(o_{t+2}, \ldots, o_T \mid q_{t+1} = j, \lambda)}_{\beta_{t+1}(j)}$$

The three simplifications each follow from the HMM independence assumptions: the transition depends only on the current state (Markov property), the emission depends only on the current state (output independence), and the remaining future observations given $q_{t+1} = j$ is exactly $\beta_{t+1}(j)$ by definition. Substituting back and summing over all next states $j$:

$$\boxed{\beta_t(i) = \sum_{j=1}^{N} \underbrace{A_{ij}}_{\text{transition } i \to j} \cdot \underbrace{B_{j, o_{t+1}}}_{\text{emit } o_{t+1} \text{ from } j} \cdot \underbrace{\beta_{t+1}(j)}_{\text{future from } j}}$$

Reading right to left: $\beta_{t+1}(j)$ is the probability of generating the rest of the sequence from state $j$ at $t+1$. Multiplying by $B_{j,o_{t+1}}$ accounts for emitting the next observation from $j$, and multiplying by $A_{ij}$ accounts for the transition from state $i$ to $j$. Summing over all possible next states $j$ gives the total probability of the future observations from state $i$ at time $t$.

### The E-Step: Computing Posterior Probabilities

The E-step computes two quantities — the expected number of times each transition and emission is used — using the forward and backward variables together.

**Why $\alpha$ and $\beta$ together?**

Note that $\alpha_t(i) \cdot \beta_t(i)$ gives the joint probability of all observations and being in state $i$ at time $t$. To see why, write out the product explicitly:

$$\alpha_t(i) \cdot \beta_t(i) = P(o_1, \ldots, o_t, q_t = i \mid \lambda) \cdot P(o_{t+1}, \ldots, o_T \mid q_t = i, \lambda)$$

At first glance it may seem unclear why these two factors simply multiply. The justification is the **chain rule of probability**, $P(A \cap B) = P(A) \cdot P(B \mid A)$, applied with $A = \{o_1, \ldots, o_t, q_t = i\}$ and $B = \{o_{t+1}, \ldots, o_T\}$:

$$P(o_1,\ldots,o_T, q_t=i \mid \lambda) = P(o_1,\ldots,o_t, q_t=i \mid \lambda) \cdot P(o_{t+1},\ldots,o_T \mid o_1,\ldots,o_t, q_t=i, \lambda)$$

The second factor still contains the full past history $o_1, \ldots, o_t$ in the conditioning set. But the Markov property tells us that given the current state $q_t = i$, the future observations are independent of the past observations — the current state encodes everything relevant about the past. So the history drops out:

$$P(o_{t+1},\ldots,o_T \mid o_1,\ldots,o_t, q_t=i, \lambda) = P(o_{t+1},\ldots,o_T \mid q_t=i, \lambda) = \beta_t(i)$$

$$P(o_1, \ldots, o_T, q_t = i \mid \lambda) = P(o_1, \ldots, o_t, q_t = i \mid \lambda) \cdot P(o_{t+1}, \ldots, o_T \mid q_t = i, \lambda) = \alpha_t(i) \cdot \beta_t(i)$$

In other words: $\alpha_t(i)$ carries all the information from the left side of the sequence, $\beta_t(i)$ carries all the information from the right side, and they are independent of each other given the state $q_t = i$ at the junction. Their product is therefore the full joint probability of all observations and state $i$ at time $t$.

**Define $\gamma_t(i)$** — the posterior probability of being in state $i$ at time $t$:

$$\gamma_t(i) = P(q_t = i \mid \mathbf{o}, \lambda) = \frac{P(\mathbf{o}, q_t = i \mid \lambda)}{P(\mathbf{o} \mid \lambda)} = \frac{\alpha_t(i) \cdot \beta_t(i)}{\sum_{j=1}^{N} \alpha_t(j) \cdot \beta_t(j)}$$

The denominator normalizes so that $\sum_i \gamma_t(i) = 1$ at every time step. We use the sum $\sum_j \alpha_t(j)\beta_t(j) = P(\mathbf{o} \mid \lambda)$ as the normalizer, which we already computed from the forward pass.

**Define $\xi_t(i,j)$** — the posterior probability of being in state $i$ at time $t$ *and* transitioning to state $j$ at time $t+1$:

$$\xi_t(i,j) = P(q_t = i, q_{t+1} = j \mid \mathbf{o}, \lambda)$$

Expanding using the joint distribution:

$$= \frac{P(q_t = i, q_{t+1} = j, \mathbf{o} \mid \lambda)}{P(\mathbf{o} \mid \lambda)}$$

The numerator factors using the same logic as the forward recursion. We split the joint probability at the transition between $t$ and $t+1$, applying the chain rule and then the HMM independence assumptions:

$$P(q_t = i, q_{t+1} = j, \mathbf{o} \mid \lambda)$$
$$= \underbrace{P(o_1,\ldots,o_t, q_t=i \mid \lambda)}_{\alpha_t(i)} \cdot \underbrace{P(q_{t+1}=j \mid q_t=i)}_{A_{ij}} \cdot \underbrace{P(o_{t+1} \mid q_{t+1}=j)}_{B_{j,o_{t+1}}} \cdot \underbrace{P(o_{t+2},\ldots,o_T \mid q_{t+1}=j)}_{\beta_{t+1}(j)}$$

The four factors are: the forward variable (past up to $t$), the transition, the emission at $t+1$, and the backward variable (future from $t+1$). Each simplification follows from the same Markov and output independence assumptions used throughout. So:

$$\boxed{\xi_t(i,j) = \frac{\alpha_t(i) \cdot A_{ij} \cdot B_{j, o_{t+1}} \cdot \beta_{t+1}(j)}{\sum_{i'=1}^{N} \sum_{j'=1}^{N} \alpha_t(i') \cdot A_{i'j'} \cdot B_{j', o_{t+1}} \cdot \beta_{t+1}(j')}}$$

### The M-Step: Re-estimating Parameters

Now we use $\gamma$ and $\xi$ to update the parameters. The intuition: these quantities represent **soft counts** — fractional expected usage of each parameter under the current model. The M-step sets each parameter proportional to its expected usage count.

**Re-estimating $\boldsymbol{\pi}$:**

$\pi_i$ should be the expected probability of starting in state $i$, which is simply $\gamma_1(i)$:

$$\hat{\pi}_i = \gamma_1(i)$$

**Re-estimating $\mathbf{A}$:**

The new transition probability $\hat{A}_{ij}$ is the expected number of transitions from $i$ to $j$, divided by the expected number of transitions out of $i$:

$$\hat{A}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

The numerator sums $\xi_t(i,j)$ over all time steps — this is the expected total count of the $i \to j$ transition. The denominator sums $\gamma_t(i)$ over all time steps — this is the expected total time spent in state $i$ (from which any transition must occur). Their ratio is the expected fraction of departures from $i$ that go to $j$.

**Re-estimating $\mathbf{B}$:**

The new emission probability $\hat{B}_{ik}$ is the expected number of times state $i$ emits symbol $k$, divided by the expected total time in state $i$:

$$\hat{B}_{ik} = \frac{\sum_{t=1}^{T} \gamma_t(i) \cdot \mathbf{1}[o_t = k]}{\sum_{t=1}^{T} \gamma_t(i)}$$

where $\mathbf{1}[o_t = k]$ is the indicator function — it equals 1 when the observation at time $t$ is symbol $k$, and 0 otherwise. The numerator counts the expected number of times we were in state $i$ and emitted $k$; the denominator is the total expected time in state $i$.

### Summary of Baum-Welch

The full algorithm iterates:

1. **Initialize** $\lambda = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$ randomly (or with a heuristic)
2. **E-step**: Run the forward and backward passes to compute $\alpha_t(i)$, $\beta_t(i)$, $\gamma_t(i)$, $\xi_t(i,j)$ for all $t, i, j$
3. **M-step**: Update $\hat{\boldsymbol{\pi}}$, $\hat{\mathbf{A}}$, $\hat{\mathbf{B}}$ using the formulas above
4. **Repeat** steps 2–3 until convergence (i.e., $P(\mathbf{o} \mid \lambda)$ stops increasing meaningfully)

Each iteration is guaranteed not to decrease $P(\mathbf{o} \mid \lambda)$. The proof of this guarantee is the general EM convergence proof — that is the subject of the companion document.

### Extension: Training on Multiple Sequences

The derivation above assumes a single observation sequence $\mathbf{o}$. In practice, you have a dataset of $S$ independent sequences $\mathbf{o}^{(1)}, \mathbf{o}^{(2)}, \ldots, \mathbf{o}^{(S)}$ (e.g., many different audio clips). The goal becomes maximizing the total log-likelihood across all sequences:

$$\lambda^* = \arg\max_\lambda \sum_{s=1}^{S} \log P(\mathbf{o}^{(s)} \mid \lambda)$$

The extension is straightforward: run the forward-backward pass independently for each sequence $s$, obtaining $\gamma_t^{(s)}(i)$ and $\xi_t^{(s)}(i,j)$ for each. Then the M-step simply **sums the soft counts across all sequences** before normalizing.

For the transition matrix:

$$\hat{A}_{ij} = \frac{\sum_{s=1}^{S} \sum_{t=1}^{T_s - 1} \xi_t^{(s)}(i,j)}{\sum_{s=1}^{S} \sum_{t=1}^{T_s - 1} \gamma_t^{(s)}(i)}$$

For the emission matrix:

$$\hat{B}_{ik} = \frac{\sum_{s=1}^{S} \sum_{t=1}^{T_s} \gamma_t^{(s)}(i) \cdot \mathbf{1}[o_t^{(s)} = k]}{\sum_{s=1}^{S} \sum_{t=1}^{T_s} \gamma_t^{(s)}(i)}$$

where $T_s$ is the length of sequence $s$. The structure is identical to the single-sequence case — the only change is that the numerators and denominators accumulate evidence from all sequences before the division. This pooling is what makes parameter estimates robust: a transition $A_{ij}$ that is rarely observed in any single sequence still gets a stable estimate from the aggregate counts.

### Local Optima and Initialization Sensitivity

Baum-Welch is not a convex optimization. $P(\mathbf{o} \mid \lambda)$ can have many local maxima, and the algorithm is guaranteed only to converge to one of them — which one depends entirely on initialization.

This is a practical concern, not a theoretical footnote. Poorly initialized models can converge to degenerate solutions such as:

- **State collapse**: multiple states converge to identical emission and transition distributions, effectively reducing the model to fewer states than intended
- **Dead states**: one or more states receive near-zero $\gamma_t(i)$ for all $t$, contributing nothing and wasting model capacity
- **Permutation traps**: because states are unordered, different random seeds can produce the same model with states relabeled — not a problem in itself, but makes comparing runs tricky

**Standard remedies in practice:**

1. **Multiple random restarts** — run Baum-Welch from $K$ different random initializations and keep the solution with the highest final $\log P(\mathbf{o} \mid \lambda)$
2. **K-means initialization** — cluster the observations into $N$ groups and use the cluster statistics to initialize $\mathbf{B}$, giving the model a sensible starting point
3. **Deterministic annealing** — flatten the posterior distributions early in training (making E-step assignments less committal) and sharpen them gradually, helping the model escape shallow local optima

This sensitivity is one of the core reasons modern sequence models moved toward neural approaches: a neural network trained with backpropagation on a supervised objective avoids the unsupervised local-optima problem entirely (though of course introduces its own optimization challenges).

### Tied States and Parameter Sharing

A practical technique for combating both sparse data and state collapse is **parameter tying** — constraining two or more states to share the same emission parameters. For example, in a speech recognition system, two states that both model the same phoneme in slightly different phonetic contexts might be constrained to use identical $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ values. This reduces the number of free parameters the M-step must estimate, making each estimate more reliable when training data is limited.

Formally, a set of states $\mathcal{S}$ is **tied** if:

$$\boldsymbol{\mu}_i = \boldsymbol{\mu}_j \quad \text{and} \quad \boldsymbol{\Sigma}_i = \boldsymbol{\Sigma}_j \quad \forall \, i, j \in \mathcal{S}$$

The M-step update for tied parameters simply pools the soft counts across all tied states before computing the shared estimate:

$$\hat{\boldsymbol{\mu}}_{\mathcal{S}} = \frac{\sum_{i \in \mathcal{S}} \sum_{t=1}^{T} \gamma_t(i) \cdot \mathbf{o}_t}{\sum_{i \in \mathcal{S}} \sum_{t=1}^{T} \gamma_t(i)}$$

The connection to modern deep learning is direct: **weight sharing** in neural networks is the same idea applied at massive scale. A convolutional filter applied across all positions of an image is a tied parameter — the same weights are reused everywhere. An embedding table in an NLP model ties all positions that refer to the same word token to the same vector. The HMM case is simply a discrete, manually specified version of what neural architectures encode structurally.

### Alternative Training Objective: Maximum Mutual Information (MMI)

Baum-Welch maximizes $P(\mathbf{o} \mid \lambda)$ — the likelihood of the observations under the model. This is a **generative** objective: it trains the model to be a good generator of the data. But for recognition tasks, we often care about something different: given multiple competing models $\lambda_1, \lambda_2, \ldots, \lambda_V$ (one per word or class), we want the correct model to score *much higher* than all alternatives.

**The problem with maximum likelihood for discrimination:**

ML training maximizes each model's score on its own training data, independently. It does not explicitly push models apart from each other. Two models can both have high likelihoods on their respective data while being nearly indistinguishable on ambiguous test examples.

**The MMI objective:**

MMI (Maximum Mutual Information, also called discriminative training) instead maximizes the posterior probability of the correct model given the observations. For a dataset of sequences $\mathbf{o}^{(s)}$ each belonging to class $v(s)$, the MMI objective is:

$$\mathcal{F}_{\text{MMI}}(\lambda) = \sum_{s} \left[\log P(\mathbf{o}^{(s)} \mid \lambda_{v(s)}) - \log \sum_{v'} P(\mathbf{o}^{(s)} \mid \lambda_{v'}) P(\lambda_{v'})\right]$$

The $\sum_s$ applies to the entire bracketed expression — both terms are summed over all training sequences. Within each bracket: the first term is the log-likelihood of the correct model for sequence $s$ — we want this to be large. The second term is the log of the total probability of the observation under all models (the denominator of Bayes' rule) — we want this to be small, i.e., we want competing models to score poorly on sequence $s$. Together, the objective pushes each correct model's score up while pushing competing models' scores down.

To see why this is "mutual information": the mutual information between the observation $\mathbf{o}$ and the class label $v$ is $I(\mathbf{o}; v) = \sum_{v} P(v) \log P(\mathbf{o} \mid v) / P(\mathbf{o})$, which has exactly the same structure — log of the class-conditional likelihood minus log of the marginal. Maximizing this over model parameters $\lambda$ is MMI training.

**Why MMI matters for modern ML:**

The MMI objective is a direct precursor to the **contrastive objectives** that dominate modern representation learning. The InfoNCE loss used in contrastive learning (e.g., SimCLR, CLIP) has exactly the same structure: the numerator scores the correct (positive) pair, and the denominator sums over all pairs including negatives. Cross-entropy classification loss is also structurally identical: it maximizes $\log P(\text{correct class}) - \log \sum_{\text{all classes}} P(\text{class})$.

The key insight MMI contributed — that discrimination requires explicit comparison against competitors, not just maximizing individual likelihoods in isolation — is now one of the foundational principles of modern deep learning training objectives.

---

## Part 5: Continuous Emissions — The Gaussian HMM

### Why Discrete Emissions Are Often Insufficient

The emission model $\mathbf{B}$ defined in Part 1 assumes observations come from a finite alphabet of $M$ symbols. This works for text (words or characters) but fails for real-valued signals like audio waveforms, accelerometer readings, or financial returns. For these, we replace $\mathbf{B}$ with a **continuous emission density**.

The most common choice is a **multivariate Gaussian** (also called a Normal distribution) per state. For a $d$-dimensional observation $\mathbf{o}_t \in \mathbb{R}^d$:

$$P(\mathbf{o}_t \mid q_t = i) = \mathcal{N}(\mathbf{o}_t; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$

where $\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$ denotes the Gaussian probability density function evaluated at $\mathbf{x}$:

$$\mathcal{N}(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

Here $\boldsymbol{\mu}_i \in \mathbb{R}^d$ is the mean of emissions from state $i$, and $\boldsymbol{\Sigma}_i \in \mathbb{R}^{d \times d}$ is the covariance matrix — a symmetric positive definite matrix capturing how observations spread around the mean. Each state $i$ now has its own mean and covariance, replacing the row $B_{i,\cdot}$ of the discrete emission table.

### What Changes in the Algorithms

The forward, backward, Viterbi, and $\gamma$/$\xi$ computations are **structurally identical** — the only change is that wherever we wrote $B_{i, o_t}$ (a table lookup), we now evaluate $\mathcal{N}(\mathbf{o}_t; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ (a density function). The rest of the recursions carry through unchanged.

### The M-Step for Gaussian Parameters

The M-step must now re-estimate $\boldsymbol{\mu}_i$ and $\boldsymbol{\Sigma}_i$ instead of the discrete emission table. These are the standard **weighted maximum likelihood** estimates for a Gaussian, where the weights are the posterior state occupancy probabilities $\gamma_t(i)$.

**Re-estimating the mean:**

The weighted mean is the sum of observations weighted by how much time we expect to be in state $i$:

$$\hat{\boldsymbol{\mu}}_i = \frac{\sum_{t=1}^{T} \gamma_t(i) \cdot \mathbf{o}_t}{\sum_{t=1}^{T} \gamma_t(i)}$$

The denominator is the total expected time in state $i$, which normalizes the weighted sum into a proper average. This is the continuous analog of the discrete $\hat{B}_{ik}$ formula: instead of counting how often state $i$ emits symbol $k$, we compute the probability-weighted centroid of all observations.

**Re-estimating the covariance:**

The weighted covariance is the probability-weighted sum of outer products of residuals from the mean:

$$\hat{\boldsymbol{\Sigma}}_i = \frac{\sum_{t=1}^{T} \gamma_t(i) \cdot (\mathbf{o}_t - \hat{\boldsymbol{\mu}}_i)(\mathbf{o}_t - \hat{\boldsymbol{\mu}}_i)^T}{\sum_{t=1}^{T} \gamma_t(i)}$$

The outer product $(\mathbf{o}_t - \hat{\boldsymbol{\mu}}_i)(\mathbf{o}_t - \hat{\boldsymbol{\mu}}_i)^T$ is a $d \times d$ matrix capturing how observation $\mathbf{o}_t$ deviates from the state mean. Weighting by $\gamma_t(i)$ and averaging gives the expected spread of observations around the mean, weighted by how likely we are to be in state $i$ at each step.

To see why these are the correct M-step updates: in the EM framework, the M-step maximizes the expected complete-data log-likelihood $\mathbb{E}[\log P(\mathbf{o}, \mathbf{q} \mid \lambda)]$ where the expectation is over the posterior $P(\mathbf{q} \mid \mathbf{o}, \lambda)$. For a Gaussian emission, the terms involving $\boldsymbol{\mu}_i$ and $\boldsymbol{\Sigma}_i$ are:

$$\sum_{t=1}^{T} \gamma_t(i) \cdot \log \mathcal{N}(\mathbf{o}_t; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$$

Taking the gradient with respect to $\boldsymbol{\mu}_i$, setting it to zero, and solving gives exactly $\hat{\boldsymbol{\mu}}_i$ above. The same procedure for $\boldsymbol{\Sigma}_i$ gives $\hat{\boldsymbol{\Sigma}}_i$. (This is the standard weighted MLE derivation for Gaussians — the EM weights $\gamma_t(i)$ play the role of soft data ownership.)

### Gaussian Mixture Emissions

A single Gaussian per state is often too restrictive — real emission distributions can be multimodal (e.g., the same phoneme spoken by different speakers clusters around different acoustic means). The standard extension is a **Gaussian Mixture Model (GMM)** emission:

$$P(\mathbf{o}_t \mid q_t = i) = \sum_{m=1}^{M} c_{im} \cdot \mathcal{N}(\mathbf{o}_t; \boldsymbol{\mu}_{im}, \boldsymbol{\Sigma}_{im})$$

where $c_{im} \geq 0$ are mixture weights with $\sum_m c_{im} = 1$, and each state $i$ now has $M$ Gaussian components. This introduces another layer of latent variables (which mixture component generated each observation), handled by an inner EM loop nested inside Baum-Welch. GMM-HMMs were the dominant model in automatic speech recognition for decades before being displaced by deep learning.

---

## Part 6: Scaling and the Underflow Problem

### The Problem

At each step of the forward pass, $\alpha_t(i)$ is a product of $t$ probabilities, each $\leq 1$. For long sequences, this product rapidly approaches zero beyond floating-point precision. For example, with $T = 1000$ and average per-step probability $0.1$, we have $\alpha_{1000} \sim 0.1^{1000} = 10^{-1000}$, which is far below the smallest representable double ($\sim 10^{-308}$).

This is called **numerical underflow**. Without correction, all $\alpha$ values collapse to zero, making every ratio (used in $\gamma$ and $\xi$) undefined.

### The Scaling Fix

The standard solution (Rabiner, 1989) is to normalize the forward variable at each time step so it never underflows, while keeping track of the normalization constants separately.

**Define a scaling factor** $c_t$ at each step:

$$c_t = \frac{1}{\sum_{i=1}^{N} \alpha_t(i)}$$

After computing the raw $\alpha_t(i)$ values, we immediately rescale:

$$\hat{\alpha}_t(i) = c_t \cdot \alpha_t(i)$$

By construction, $\sum_i \hat{\alpha}_t(i) = 1$ at every time step — the scaled forward variable is always a proper probability distribution over states.

The same scaling factors are applied to the backward variable:

$$\hat{\beta}_t(i) = c_t \cdot \beta_t(i)$$

### What Happens to the Log-Likelihood?

We need to show that the scaling factors $c_t$ encode $P(\mathbf{o} \mid \lambda)$. To see this, trace what the scaling does to the forward variable over time.

At $t=1$: the raw $\alpha_1(i)$ is computed, then rescaled by $c_1$, giving $\hat{\alpha}_1(i) = c_1 \cdot \alpha_1(i)$.

At $t=2$: the raw $\alpha_2(i)$ is computed from $\hat{\alpha}_1$ (the already-scaled values), so the raw $\alpha_2(i)$ already has one factor of $c_1$ baked in. We then apply $c_2$, giving $\hat{\alpha}_2(i) = c_2 \cdot c_1 \cdot \alpha_2(i)$.

In general, after $t$ steps:

$$\hat{\alpha}_t(i) = \left(\prod_{s=1}^{t} c_s\right) \cdot \alpha_t(i)$$

At the final step $T$, summing over all states:

$$\sum_{i=1}^{N} \hat{\alpha}_T(i) = \left(\prod_{s=1}^{T} c_s\right) \cdot \sum_{i=1}^{N} \alpha_T(i) = \left(\prod_{s=1}^{T} c_s\right) \cdot P(\mathbf{o} \mid \lambda)$$

But $\sum_i \hat{\alpha}_T(i) = 1$ by construction (we normalize at every step). Therefore:

$$1 = \left(\prod_{s=1}^{T} c_s\right) \cdot P(\mathbf{o} \mid \lambda) \implies P(\mathbf{o} \mid \lambda) = \prod_{t=1}^{T} \frac{1}{c_t}$$

Taking logarithms:

$$\log P(\mathbf{o} \mid \lambda) = -\sum_{t=1}^{T} \log c_t$$

This is fully numerically stable — we are summing $T$ log values, each of moderate magnitude.

### Why Scaling Preserves $\gamma$ and $\xi$

The claim is that $\gamma_t(i)$ and $\xi_t(i,j)$ computed from scaled variables $\hat{\alpha}$ and $\hat{\beta}$ are identical to those computed from the unscaled versions. We show this explicitly.

**For $\gamma_t(i)$:**

Recall that $\hat{\alpha}_t(i) = C_t \cdot \alpha_t(i)$ where $C_t = \prod_{s=1}^{t} c_s$ is the cumulative scaling product up to time $t$. The backward variable is scaled by the single step factor: $\hat{\beta}_t(i) = c_t \cdot \beta_t(i)$.

Therefore:

$$\hat{\alpha}_t(i) \cdot \hat{\beta}_t(i) = C_t \cdot \alpha_t(i) \cdot c_t \cdot \beta_t(i) = (C_t \cdot c_t) \cdot \alpha_t(i) \cdot \beta_t(i)$$

The same factor $(C_t \cdot c_t)$ appears in the denominator when we sum over all states:

$$\sum_{j=1}^{N} \hat{\alpha}_t(j) \cdot \hat{\beta}_t(j) = (C_t \cdot c_t) \cdot \sum_{j=1}^{N} \alpha_t(j) \cdot \beta_t(j)$$

Forming the ratio:

$$\gamma_t(i) = \frac{\hat{\alpha}_t(i) \cdot \hat{\beta}_t(i)}{\sum_j \hat{\alpha}_t(j) \cdot \hat{\beta}_t(j)} = \frac{(C_t \cdot c_t) \cdot \alpha_t(i) \cdot \beta_t(i)}{(C_t \cdot c_t) \cdot \sum_j \alpha_t(j) \cdot \beta_t(j)} = \frac{\alpha_t(i) \cdot \beta_t(i)}{\sum_j \alpha_t(j) \cdot \beta_t(j)}$$

The scaling factor cancels exactly. $\gamma_t(i)$ is unchanged.

**For $\xi_t(i,j)$:**

The numerator of $\xi_t(i,j)$ involves $\hat{\alpha}_t(i) \cdot A_{ij} \cdot B_{j,o_{t+1}} \cdot \hat{\beta}_{t+1}(j)$. Substituting:

$$\hat{\alpha}_t(i) \cdot A_{ij} \cdot B_{j,o_{t+1}} \cdot \hat{\beta}_{t+1}(j) = C_t \cdot \alpha_t(i) \cdot A_{ij} \cdot B_{j,o_{t+1}} \cdot c_{t+1} \cdot \beta_{t+1}(j)$$

Since $C_t \cdot c_{t+1} = \prod_{s=1}^{t+1} c_s = C_{t+1}$, this equals $C_{t+1} \cdot \alpha_t(i) \cdot A_{ij} \cdot B_{j,o_{t+1}} \cdot \beta_{t+1}(j)$.

The same factor $C_{t+1}$ appears in the denominator (the double sum over all $i', j'$), so it cancels identically. $\xi_t(i,j)$ is also unchanged.

This is why scaling is the correct practical solution: it prevents underflow while leaving all the quantities we actually care about — the posterior probabilities used in the M-step — completely intact.

---

## Part 7: Modern ML Connections

HMMs are not just a historical curiosity. Their architecture, algorithms, and failure modes directly shaped or foreshadowed several core ideas in modern deep learning.

### Connection 1: HMMs as a Special Case of Latent Variable Models

An HMM is a **latent variable model** — a model with observed variables $\mathbf{o}$ and unobserved (latent) variables $\mathbf{q}$. The parameters are estimated by maximizing the marginal likelihood $P(\mathbf{o} \mid \lambda)$, which requires integrating out the latent variables.

This is exactly the structure of **Variational Autoencoders (VAEs)**. In a VAE:

- The encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ is the analog of the E-step posterior — it infers a distribution over latent variables given observations
- The decoder $p_\theta(\mathbf{x} \mid \mathbf{z})$ is the analog of the emission model $B$
- The prior $p(\mathbf{z})$ is the analog of the initial state and transition distributions

The key difference is that VAEs use **amortized inference** (the encoder is a neural network that generalizes across data points) rather than exact EM (which re-runs inference from scratch for each sequence). The underlying objective — maximize a lower bound on $\log P(\mathbf{x})$ — is the same ELBO (Evidence Lower Bound) that EM implicitly optimizes. We will derive the ELBO explicitly in the EM companion document.

### Connection 2: The Forward Algorithm as a Special Case of Belief Propagation

The forward recursion:

$$\alpha_{t+1}(j) = \left[\sum_{i=1}^{N} \alpha_t(i) \cdot A_{ij}\right] \cdot B_{j, o_{t+1}}$$

is a specific instance of the **sum-product algorithm** (also called belief propagation) running on a chain-structured graphical model. At each step, we are passing a **message** — a vector of probabilities over states — from time $t$ to time $t+1$.

This generalizes to arbitrary graphical models: on trees, belief propagation is exact and has the same $O(N^2)$-per-edge complexity. On graphs with cycles it becomes **loopy belief propagation**, which is approximate but widely used.

The message-passing perspective is also directly analogous to how **recurrent neural networks (RNNs)** operate. An RNN hidden state $\mathbf{h}_t$ is the continuous, learned analog of the discrete forward variable $\alpha_t$: it summarizes all past inputs into a vector that is propagated forward in time. The key limitation of both — that the message is a fixed-size vector and older information gets compressed — is what motivated the development of attention mechanisms.

### Connection 3: Viterbi as Dynamic Programming on a Sequence — Precursor to CRFs

The Viterbi algorithm is dynamic programming on the trellis (the graph of states over time). Its structure — maximizing a sum of local scores along a path — is exactly the inference procedure used in **Conditional Random Fields (CRFs)**.

A CRF is a discriminative model that directly parameterizes $P(\mathbf{q} \mid \mathbf{o})$ (rather than the joint $P(\mathbf{o}, \mathbf{q})$ as HMMs do). In sequence labeling (named entity recognition, part-of-speech tagging), the standard architecture is a **BiLSTM-CRF**: a bidirectional LSTM produces per-token feature vectors, and a CRF layer on top runs Viterbi to find the globally optimal label sequence. The Viterbi algorithm you derived above runs unchanged inside these models today.

### Connection 4: Baum-Welch as EM — and the Bridge to Variational Inference

Baum-Welch is EM applied to the HMM. The E-step computes exact posteriors over hidden states; the M-step maximizes the expected complete-data log-likelihood.

This exact inference is only tractable because the HMM has **chain structure** — the trellis has no cycles, so the forward-backward algorithm computes exact posteriors in polynomial time.

For more complex latent variable models — deep generative models, topic models with non-conjugate priors — exact E-steps are intractable. The modern solution is **variational inference**: replace the exact posterior $P(\mathbf{q} \mid \mathbf{o})$ with a parameterized approximation $q_\phi(\mathbf{q})$ and optimize $\phi$ to minimize KL divergence. This yields the ELBO objective used in VAEs. The HMM is thus the "exact inference" anchor point from which the entire family of approximate inference methods departs.

### Connection 5: Attention as a Soft, Differentiable Forward Algorithm

Perhaps the deepest connection. In the forward-backward algorithm, $\gamma_t(i) = P(q_t = i \mid \mathbf{o}, \lambda)$ is a **soft assignment** of time step $t$ to each state $i$ — rather than committing to a single state (as Viterbi does), it holds a probability distribution over all states simultaneously. This is exactly the structure of attention: instead of hard-routing each token to a single "state," attention computes a weighted combination over all possible context positions.

More concretely: in early sequence-to-sequence models with attention (Bahdanau et al., 2015), the decoder at each output step computes an **alignment weight** $a_{ts}$ over each encoder position $s$. This weight plays the same role as $\gamma_t(i)$ — it asks "how relevant is position $s$ (state $i$) to generating output at step $t$?" The context vector passed to the decoder is then a weighted sum of encoder states, exactly as the forward variable computes a weighted mixture over hidden states.

The **attention mechanism** in Transformers extends this further:

1. The "states" are replaced by learned key/value embeddings
2. The alignment weights are computed via a learned dot-product similarity ($\text{softmax}(\mathbf{Q}\mathbf{K}^T / \sqrt{d})$) rather than fixed transition probabilities
3. Every position attends to every other position simultaneously (rather than sequentially)

The connection is not merely analogical. Bahdanau et al. explicitly motivated their attention mechanism as a soft, differentiable alternative to the hard Viterbi alignment used in HMM-based speech recognition — the HMM required committing to a single alignment path, while attention allows a soft distribution over all alignments. The progression HMM → soft attention → Transformer self-attention is one of the cleanest intellectual lineages in modern deep learning.

### Connection 6: Neural HMMs — Replacing $\mathbf{B}$ with a Neural Network

The most direct bridge from HMMs to modern deep learning is to keep the HMM's probabilistic structure (hidden states, transitions, forward-backward inference) but **replace the emission model with a neural network**.

In a **Neural HMM**, the emission probability for state $i$ at time $t$ is parameterized as:

$$P(\mathbf{o}_t \mid q_t = i) = f_\theta(\mathbf{o}_t, i)$$

where $f_\theta$ is a neural network with parameters $\theta$. Rather than a fixed Gaussian or a lookup table, the network can model arbitrarily complex, high-dimensional emission distributions. Training alternates between:

- **E-step**: Run forward-backward with the current neural emission probabilities to compute $\gamma_t(i)$ and $\xi_t(i,j)$ — this is unchanged from standard Baum-Welch
- **M-step**: Update the transition matrix using the soft-count formulas, and update $\theta$ by gradient descent on the weighted negative log-likelihood:

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \sum_{i=1}^{N} \underbrace{\gamma_t(i)}_{\text{soft weight: how much} \atop \text{state } i \text{ owns step } t} \cdot \underbrace{\log f_\theta(\mathbf{o}_t, i)}_{\text{log-likelihood of } \mathbf{o}_t \atop \text{under state } i\text{'s neural emission}}$$

Reading term by term: $\log f_\theta(\mathbf{o}_t, i)$ is how well the neural emission model for state $i$ explains the observation at time $t$ — we want this to be large (close to zero) for the right states. Multiplying by $\gamma_t(i)$ weights each state's contribution by how much the E-step believes state $i$ is responsible for time $t$. Summing over all $t$ and $i$ and negating gives a loss to minimize: states that the E-step assigns high responsibility are pushed to explain their assigned observations better.

This is still an EM algorithm — the E-step provides the weights $\gamma_t(i)$, and the M-step optimizes the neural emission model using those weights as a soft supervision signal.

**Why this matters for modern ML:** Neural HMMs are the conceptual midpoint between classical HMMs and fully neural sequence models. They preserve the **structured latent variable** interpretation (there are discrete hidden states with Markovian dynamics) while gaining the representational power of neural networks for the observations. This architecture underlies modern neural speech synthesis systems (e.g., the original Tacotron used an attention mechanism that can be interpreted as a Neural HMM alignment model) and remains an active research area for tasks requiring interpretable discrete structure alongside neural expressiveness.

The further step — replacing the transition matrix $\mathbf{A}$ with a neural network as well — leads to **recurrent neural networks**: the hidden state becomes continuous, the transition becomes a learned nonlinear function, and the emission is a neural decoder. At that point the HMM has been fully neuralized into an RNN.

### Summary of Connections

| HMM Concept | Modern ML Analog |
|---|---|
| Hidden state sequence | Latent variables in VAEs, RNN hidden states |
| Left-right topology ($A_{ij}=0$ for $j<i$) | Causal masking in autoregressive Transformers |
| Discrete emission table $\mathbf{B}$ | Gaussian / GMM emissions; neural emission networks |
| Tied state parameters | Weight sharing in CNNs; shared embeddings in NLP |
| Forward algorithm | Sum-product / belief propagation; RNN forward pass |
| Posterior decoding ($\gamma$) | Soft attention weights in seq2seq models |
| Viterbi algorithm | CRF decoding; beam search in seq2seq models |
| Baum-Welch / EM | Variational inference; ELBO objective in VAEs |
| MMI discriminative training | Contrastive loss (SimCLR, CLIP); cross-entropy classification |
| Transition matrix $\mathbf{A}$ | Learned attention weights in Transformers |
| Local optima / init sensitivity | Motivation for supervised neural sequence models |
| Neural HMM (neural $\mathbf{B}$) | Direct precursor to RNNs and neural TTS systems |

---

## Key Takeaways

1. An HMM models sequences via two independence assumptions: the Markov property on hidden states, and output independence of observations given states
2. **Model topology** is a structural choice: ergodic models allow any transition; left-right models constrain $A_{ij} = 0$ for $j < i$, encoding temporal directionality — the HMM analog of causal masking in Transformers
3. The **forward algorithm** evaluates $P(\mathbf{o} \mid \lambda)$ in $O(N^2 T)$ using dynamic programming — replacing exponential enumeration with incremental message passing
4. The **Viterbi algorithm** decodes the most likely state sequence in $O(N^2 T)$ by replacing the sum in the forward recursion with a max and storing backpointers; **posterior decoding** is an alternative that maximizes per-position accuracy using $\gamma_t(i)$ instead, and the two can disagree
5. The **Baum-Welch algorithm** learns parameters via EM: the E-step computes soft state occupancy posteriors ($\gamma$, $\xi$) using forward-backward; the M-step re-estimates parameters as normalized expected counts summed across all training sequences
6. Baum-Welch is non-convex and highly sensitive to initialization — multiple restarts or K-means initialization are standard practice; **tied states** reduce the parameter count by sharing emission parameters across states, the HMM analog of weight sharing in neural networks
7. **MMI training** replaces the generative ML objective with a discriminative one — maximizing the correct model's score relative to all competitors — a direct conceptual precursor to contrastive losses and cross-entropy in modern deep learning
8. **Continuous (Gaussian) emissions** replace the discrete table $\mathbf{B}$ with per-state Gaussians parameterized by $\boldsymbol{\mu}_i$ and $\boldsymbol{\Sigma}_i$; the M-step updates become probability-weighted MLE estimates; GMM emissions extend this to multimodal per-state distributions
9. **Numerical scaling** is essential in practice — normalizing $\alpha_t$ at each step and tracking log-scale normalization constants prevents underflow without affecting the posteriors
10. HMMs are the tractable, exact-inference anchor point for a family of ideas that extends directly to RNNs, CRFs, VAEs, Neural HMMs, and Transformer attention
