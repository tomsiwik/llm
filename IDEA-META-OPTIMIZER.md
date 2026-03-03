# IDEA: Meta-Optimizer Framework — Candidate Generation as a Design Space

## Core Insight

We accidentally built something more general than "evolutionary gradient training."
The real contribution is a **two-level optimization architecture**:

```
Level 1 (Inner): Generate candidate weight updates from gradient information
Level 2 (Outer): Evaluate candidates on actual multi-task loss, select best
```

All prior continual learning optimizers (GEM, A-GEM, OGD, PCGrad, CAGrad)
operate in **gradient space** — they analytically modify the gradient direction
before applying it. Our approach operates in **weight space** — it evaluates
the actual resulting weights. This is more expensive but more principled:
gradient-space methods assume local linearity (the gradient predicts loss
change), which breaks down at large learning rates, sharp loss landscapes,
and near task boundaries.

The current candidates {full step, half step, 1.5x, mutated} are ad-hoc.
But the framework supports ANY candidate generation strategy — making this
a **meta-optimizer** where the candidate generator is a pluggable component.

## Why This Matters

The stability-plasticity tradeoff in continual learning is fundamentally a
**multi-objective optimization problem**. Current methods commit to a single
point on the Pareto frontier at design time (EWC chooses lambda, GEM chooses
memory size). Our framework lets the optimizer **discover the right tradeoff
per step** based on the actual loss landscape, not a fixed hyperparameter.

## The Design Space

### Axis 1: Step Size Candidates

Current: {0.5x, 1.0x, 1.5x} — three discrete points.

Better strategies:
- **Continuous line search**: evaluate at {0.25, 0.5, 0.75, 1.0, 1.25, 1.5}
  along the gradient direction. This is a 1D line search — cheap and
  well-understood, but now with multi-task evaluation.
- **Adaptive range**: track which step size was selected over the last K steps.
  If half-step wins consistently, narrow the range to {0.3, 0.5, 0.7}.
  If full step wins, widen to {1.0, 1.5, 2.0}. This is step-size adaptation
  driven by multi-task fitness, not gradient statistics.
- **Per-parameter step sizes**: different learning rates for expert weights
  vs attention weights vs embeddings. The outer evaluator selects the
  combination, not just a scalar multiplier.

### Axis 2: Direction Candidates

Current: gradient direction + random perturbation.

Better strategies:
- **Fisher-informed stepping**: the Fisher information matrix tells us which
  parameter directions matter most for old tasks. Candidates that step
  orthogonal to high-Fisher directions preserve old knowledge; candidates
  that step along low-Fisher directions change "unimportant" parameters.

  ```
  Candidate A: step along gradient (standard)
  Candidate B: step along gradient projected onto low-Fisher subspace
  Candidate C: step along gradient projected onto null space of old-task Hessian
  ```

  The outer evaluator picks whichever actually works best. This combines
  the analytical insight of OGD with the empirical verification of our method.

- **Momentum-diversity**: generate candidates from different optimizer states:
  ```
  Candidate A: Adam update (adaptive, momentum-based)
  Candidate B: SGD update (raw gradient, no momentum)
  Candidate C: Sign-SGD update (sign of gradient only — robust to outliers)
  Candidate D: Previous step's update direction (momentum continuation)
  ```
  Each optimizer has different inductive biases about which updates are good.
  The outer evaluator empirically selects the best one per step.

- **Gradient decomposition**: decompose the gradient into components:
  ```
  g = g_shared + g_expert_specific
  Candidate A: apply both (full gradient)
  Candidate B: apply only g_expert_specific (protect shared params)
  Candidate C: apply only g_shared (protect expert params)
  Candidate D: apply g_shared + 0.5 * g_expert_specific (hybrid)
  ```

### Axis 3: Subset Candidates

Current: all trainable parameters are updated together.

Better strategy: **parameter-group selective updates**:
```
Candidate A: update everything (standard)
Candidate B: update only expert MLP weights (freeze attention)
Candidate C: update only attention (freeze experts)
Candidate D: update only the expert that was most activated this step
```

This lets the optimizer discover that "on this particular step, updating
attention hurts old task performance, so only update the experts" —
a dynamic, per-step version of selective freezing.

### Axis 4: Evaluation Strategies

Current: evaluate on 1 old doc + 1 new doc.

Better strategies:
- **Diverse old-task sampling**: evaluate on K old docs from different old
  tasks, not just one. Reduces variance (Criticism 4 from review).
- **Exponential moving average baseline**: instead of measuring old-task loss
  fresh each step, maintain a running average. Threshold becomes more stable.
- **Proxy evaluation**: instead of full forward pass, use a cheap proxy:
  - Gradient inner product (like A-GEM, but as one of several signals)
  - Parameter distance from task-A checkpoint
  - Fisher-weighted parameter change

  Use the cheap proxy for most candidates, full eval only for the top-2.
  This reduces cost from N forward passes to 2-3.

## What Makes This a Framework Paper

The key claim: **candidate generation and multi-task evaluation are
orthogonal design dimensions**. Any combination of:

```
candidate_strategy × evaluation_strategy × selection_rule
```

yields a valid continual learning optimizer. Our experiments show:
- Simple candidates + simple evaluation already beats known methods (BWT improvement)
- The Pareto frontier shifts with different candidate/evaluation combos
- No single configuration dominates — the best choice depends on the task

This is analogous to how the Transformer paper didn't invent attention OR
feed-forward layers, but showed that the specific combination creates
something more powerful than either alone.

## Key Experiment: Prove the Framework Adds Value

Run a grid over:
- 4 candidate strategies × 3 evaluation strategies × 2 selection rules
- Each point produces a (BWT, L_B) pair
- Plot all 24 points on the Pareto frontier
- Compare against GEM, A-GEM, EWC Pareto frontiers (sweep their hyperparams)

If our frontier dominates or covers different operating points, the framework
is the contribution, not any single configuration.

## Relationship to Other Ideas

- **IDEA-EXPERT-LIFECYCLE**: the meta-optimizer handles shared parameter
  updates; the expert lifecycle handles expert-specific knowledge management.
  They operate at different levels and are complementary.
- **Population Based Training (PBT)**: PBT evolves hyperparameters across
  a population of models. Our meta-optimizer evolves weight updates within
  a single model. PBT is inter-model; ours is intra-step.
- **Neural Architecture Search (NAS)**: NAS searches over architectures;
  we search over optimizer strategies. Same meta-learning principle,
  different search space.
