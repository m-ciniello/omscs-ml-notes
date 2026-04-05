## Audience and Scope

These instructions govern the creation of rigorous, standalone ML study notes for a student in the Georgia Tech OMSCS (Online Master of Science in Computer Science) program, specializing in Machine Learning. The student has 8 years of data science experience and strong foundations in calculus, linear algebra, probability, and algorithms. The reader is comfortable with formalism and expects thorough, gap-free explanations. Each session covers one well-scoped ML topic — e.g. PCA, gradient descent, the EM algorithm — and produces a single reference document for that topic.

---

## Source Material

Each task begins with one or more source readings (lecture notes, textbook chapters, papers). Treat these as a minimum syllabus — every topic they cover must appear in the notes, but depth and quality are not bounded by the source. The readings are a starting point and a floor, not templates to transcribe from.

Where a reading is thin, poorly motivated, or weak on intuition, go deeper. If the provided readings are insufficient — shallow, unclear, or missing standard material — supplement freely from authoritative outside sources (textbooks, survey papers, official documentation) and note the source inline. When readings contradict each other, prefer the more rigorous or widely-accepted treatment and note the discrepancy.

---

## Writing Standards

### Structure

Organize each topic as: **motivation → derivation → interpretation**. The output should be standalone notes rigorous enough to serve as a complete reference without consulting external sources.

#### Section framing

Each major section should follow a natural arc: open with a brief orienting overview (what this section covers, why it matters, and how its parts connect), use clear transitions between subsections so the reader always knows where they are and what question comes next, and close with a short wrap-up that ties the key results together and previews what follows. These are structural principles, not labels — do not insert literal "Roadmap" or "Wrap-up" headings. The flow should feel like a well-structured lecture, not a fill-in-the-blank template.

#### Formatting

Key results should be visually distinct — block equations, brief result headers (e.g. **Result: the bias-variance decomposition**), and descriptive sub-headings rather than generic labels like "Example" or "Proof."

#### Cross-topic connections

Links to adjacent algorithms or later sections belong at the *end* of the current section's narrative as a forward pointer — not inserted mid-section before the section's own story is complete.

#### Extensions beyond the reading

Where natural extensions exist — modern applications, connections to adjacent ML topics, results that meaningfully deepen understanding — include them. The goal is comprehensive topic coverage; there is no need to distinguish between material that originated in the assigned readings and material added to fill gaps or deepen understanding.

---

### Intuition

Intuition is not a bonus layer added after the math — it is the *main product* of these notes. A derivation without intuition is a failed derivation, regardless of how correct it is. Rigor is a tool in service of understanding, never a substitute for it. **When in doubt, over-explain the intuition.**

#### Motivation before presentation

Before introducing an algorithm, variable, or proof strategy, establish: what problem does this solve, and why does the naïve approach fail? The math should feel like a solution to a clearly stated problem, not a result that appears from nowhere.

#### Every result: meaning, then significance

After every significant derivation or result, address two questions in sequence:

1. **What does it say?** Explain the result in plain terms. Use analogies, concrete numerical examples (a single worked example with small numbers, e.g. $D = 3$, $M = 1$, often communicates more than a paragraph of abstract description), or "what would happen if..." reasoning to make it stick.
2. **Why does it matter?** Answer "so what?" — why we derived it, what it enables, and how it connects to the larger goal. Watch for three failure modes where significance is most often dropped: (a) re-deriving a known quantity via a new route, (b) introducing a new object or algebraic reformulation whose purpose isn't yet clear — always state upfront what the new form reveals that the old one didn't, and (c) a clean derivation whose payoff is implicit. In all three cases, add a bridging paragraph before moving on.

---

### Rigor

Rigor means the reader can follow every step independently — not just verify that the answer is correct, but understand why each manipulation was made and what it achieves.

#### Derivations

When introducing an important result, bound, or formula, derive it step by step and treat each manipulation as if the reader cannot fill in gaps themselves. Every non-trivial algebraic step should be shown explicitly — do not skip steps with phrases like "it can be shown that" or "expanding gives us." Name the technique being used at each step (e.g. "telescoping", "applying the quadratic formula", "using Vieta's formulas") and explain why it is valid. If a derivation is genuinely out of scope, say so explicitly and explain why.

If a reading's derivation is technically correct but unclear or poorly structured, replace it with a cleaner version from standard references, first principles, or an alternative proof strategy. The goal is the most illuminating explanation of the result, not a faithful transcription of the source.

#### Proof style

Prefer single-direction proofs: start from one side of an identity and derive the other, rather than simplifying both sides independently and noting they match. If a two-sided approach is unavoidable, explicitly label which side is being worked on at each stage. In either style, call out the most non-obvious manipulation by name — don't bury it inside a compound equation.

#### Terminology and claims

When introducing technical or mathematical terminology (e.g. "dimensionless", "spectral radius", "unbiased estimator"), always define it inline with a concrete example. Use judgment — common terms like "gradient" or "eigenvalue" don't need definition, but anything that has a precise mathematical meaning beyond everyday usage should be explained.

When making a claim (e.g. "the inverse Hessian is the ideal preconditioner", "the gradient inequality holds"), always justify it. If the full proof is out of scope, explicitly say so and provide the intuition instead. Never present a claim as self-evident when it isn't.

#### Notation and final results

When presenting a key formula as a final result, annotate each term (e.g. using underbraces) and follow it with a prose walkthrough explaining what each term represents and how to read the expression. When introducing notation with non-obvious scope or reading conventions (e.g. argmax, summation operators), explain the convention inline with a concrete example.

---

### Visuals and Code

Figures and code are not optional extras — they are *primary* tools for building intuition, on equal footing with prose and algebra. A well-chosen figure or a 20-line code snippet often communicates more than a page of description. **Actively look for opportunities to include them**, rather than treating them as a last resort.

#### Figures

Add a figure whenever a concept has a geometric or visual dimension: transformations, probability contours, algorithm convergence, cluster geometry, decision boundaries, data flow, side-by-side method comparisons. During writing, place a placeholder tag at the right location — do not interrupt writing to search for or download images. Use one of two tag formats:

- **From the readings:** `[FIG:READING — Figure X.X from [Source]: brief description of what it shows]`
- **Not in the readings:** `[FIG:ORIGINAL — precise description of what to search for, including algorithm name, what the figure should show, and why it helps]`

The description in `[FIG:ORIGINAL]` tags must be specific enough to drive a good search — not "a figure of k-means" but "k-means convergence showing how centroid positions shift across three iterations on a two-cluster dataset."

#### Code snippets

A short, self-contained code example is one of the most effective ways to make an abstract concept concrete. Use them generously — the bar is *"would seeing this generated or computed help the reader's intuition?"* Common high-value cases include:

- **Visualizing a key concept** — e.g. the ICA identifiability example in the PCA/ICA notes: a 20-line scatter plot showing uniform sources (square vs. diamond) alongside Gaussian sources (circle vs. circle) makes the identifiability theorem visceral in a way that no prose description can. Any time a mathematical distinction (identifiability, convergence, separability) can be *shown* as a picture generated from a few lines of code, do it.
- **Iterative algorithms where state change over time is the insight** — e.g. watching k-means centroids shift across iterations, or EM responsibilities redistribute. A static figure shows one snapshot; code lets the reader trace the sequence.
- **Counterintuitive numerical behaviour** — e.g. demonstrating that nearest-neighbour distances concentrate in high dimensions, or that a naïve log-likelihood computation underflows where a log-sum-exp formulation does not. The surprise only lands when you see actual numbers.
- **Non-trivial gap between theory and implementation** — e.g. numerical stability tricks, edge cases, or parameter sensitivity that would take several paragraphs to describe but is obvious in ten lines.

Do not add code simply to transliterate a formula into Python — if the derivation is clear and there's nothing new to *see*, code adds noise. When including a snippet, keep it short (≤ 30 lines), self-contained, and focused on the insight, not a full implementation. Embed inline as a fenced Python code block.

---

### Attribution

When a specific derivation or framing is drawn from a particular source, note it inline (e.g. "following Wasserman §9.13") so the reader knows where to find the full treatment. At the end of each document, include a **Sources and Further Reading** section listing the assigned readings and any key outside sources consulted.

---

## Session Workflow

### Reviews

After adding substantive content, do a targeted review of the surrounding context: check for contradictions, undefined terms, and missing "so what" moments. Reserve full document reviews for when explicitly requested. Do not perform unsolicited broad reviews after minor edits.

### Figure resolution

Do not resolve figure tags until the student explicitly asks (e.g. "let's resolve figures"). When asked, resolve all unresolved tags in one batch:

1. **Search for `[FIG:ORIGINAL]` figures.** For each, search for an existing figure from a reputable source (scikit-learn docs, Wikipedia, the original paper, a standard textbook). If found, download it into `images/`, replace the tag with a markdown image, and add a caption with source attribution. If not found, mark the tag as searched and carry it to step 2.
2. **Compile remaining tags.** Present the student with a single list of all unresolved figures — all `[FIG:READING]` tags (which require screenshots from the source) and any `[FIG:ORIGINAL]` that couldn't be found — and ask the student to save them into the `images/` folder.
3. **Finalize.** Once the student confirms the files are in `images/`, rename each to a descriptive `snake_case` filename, replace its placeholder tag with a markdown image, and add a caption.

**File conventions.** Each topic folder has an `images/` subfolder. All figures are embedded using relative paths (`![caption](images/filename.png)`) and must have a one-sentence italic caption stating what the figure shows and attributing its source.
