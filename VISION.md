# Vision: Contextual Expert Routing via Composable LoRA Specialists

## The End Goal

Beat larger monolithic models through focused specialization. A 0.5B base model
augmented with N tiny LoRA experts, each a domain specialist, should match or
exceed a 1.5B model — at a fraction of active parameters per token.

## The Core Insight

A LoRA adapter `ΔW = A @ B` naturally separates into two roles:

- **A (d_in, r)**: The receptive field. Determines *what* the expert responds to.
  `||x @ A.T||` measures how well input `x` matches this expert.
- **B (r, d_out)**: The knowledge. Transforms the matched signal into output.

This means **routing is free**. The same A matrix that computes the LoRA
projection also tells you whether this expert should fire. No external router
needed. Each expert is *self-describing*.

## The Analogy

Expert routing IS token prediction over a vocabulary of capabilities.

- Token vocabulary: 50K entries, each a fixed embedding. Input selects the
  closest embedding via dot product (attention).
- Expert vocabulary: N entries, each a learned receptive field (A matrix).
  Input selects the closest expert via `||x @ A.T||`.

Adding a new word = adding an embedding. Adding a new expert = storing (A, B).
Both are zero-retraining operations.

## The Decomposition Principle

When experts share knowledge (e.g., all code experts know variable syntax),
Procrustes alignment can separate:

- **Shared knowledge**: Common across experts. Applied always.
- **Unique knowledge**: Domain-specific residuals. Routed contextually.

This eliminates the dilution problem of naive merging (averaging N experts at
1/N strength). Shared knowledge stays at full strength. Unique knowledge
activates only when relevant, also at full strength.

## The Protocol

1. **Train independently**: Each expert trains on its domain. Standard LoRA.
2. **Decompose**: Procrustes-align, extract shared + unique components.
3. **Register**: Store (A, B) pairs in a library. One library per layer.
4. **Route**: At inference, `||x @ A_i.T||` scores each expert per token.
   Top-k selection. Full-strength application. No retraining.

Adding expert N+1 = train on new domain, register (A, B). Done.

## What We've Proven

From the 13-method composition benchmark on Qwen2.5-Coder-0.5B:

- **LoRA deltas are naturally orthogonal** (cosine ≈ 0.000 between Python and
  JavaScript adapters). Independent training produces non-interfering experts.
- **Merging dilutes**: Task arithmetic at λ=0.5 halves each expert's strength.
  TIES, DARE, SVD, Procrustes — all lose specialization through averaging.
- **Routing beats merging**: CAT (both adapters active) and Router (learned
  per-token selection) preserve both domains at full quality.
- **The A matrix IS the routing key**: Self-routing via `||x @ A.T||` is the
  natural mechanism. Already proven in `lora_atom.py` for continual learning.

## What We Learned: A Matrices Don't Self-Route

Experiments 1-3 revealed that **standard LoRA A matrices do NOT self-route**:
- 3 scoring methods (raw, normalized, full delta) all give ~50% accuracy — coin flip
- A matrices learn "useful projections" generically, not domain-discriminative ones
- `||x @ A||²` is dominated by input norm `||x||²`, not by domain alignment
- LoRA training optimizes A for reconstruction quality, not for discrimination

**Root cause**: Routing and computation are fundamentally different tasks.
Coupling them in a single matrix forces a compromise that satisfies neither.

## What We Learned: Contrastive Routing Keys Fail at Micro Scale

Experiment 1b tested InfoNCE-trained routing keys K_i to replace softmax
router calibration. **All three kill thresholds exceeded:**

- Routing accuracy: 53.3% (target >85%, kill <70%) — barely above random
- Composition quality: +141% worse than joint (target <5%, kill >10%)
- Linear probe beats contrastive: 59.8% vs 53.3% — contrastive loss adds nothing
- Tau sweep (0.05-1.0): no temperature helps

**Root cause**: MATH.md Assumption 6 falsified — at d=64 with a-m vs n-z
character-level tokenization, domains are NOT distinguishable in hidden state
space. Even a linear probe only gets ~60%. The character-level representations
don't carry enough domain-discriminative signal.

**Key insight: task-routing > identity-routing.** The softmax router works
(+0.2% vs joint) because it routes by reconstruction quality — which groups
minimize prediction error — not by domain identity. At micro scale with
similar domains, task-aligned routing (reconstruction loss) dominates
identity-aligned routing (contrastive loss).

**Not necessarily dead at macro scale**: with larger models (d=256+), real
domains (Python vs JavaScript), and BPE tokenization creating domain-specific
tokens, hidden states will carry much stronger domain signal. Deferred to
macro validation.

## The Routing Story So Far

1. **A-matrix self-routing**: Dead. A matrices don't discriminate (~50%).
2. **Contrastive keys (K_i)**: Dead at micro scale. Domains indistinguishable at d=64.
3. **Softmax router calibration**: Works (+0.2% vs joint, ~100 steps mixed data).
   Routes by task quality, not domain identity. Validated baseline for composition.
4. **Sparse routing (top-1)**: Dead at micro scale. Phase transition between k=1
   (+200% degradation) and k=2 (within 1.6%). Hard selection amplifies flat
   probability distribution — C_1=0.285 means 71% of information silenced.
   k=2 validated as optimal sparsity. Capacity-bound (8K params/group), not
   mechanism-bound — Switch Transformer uses k=1 at scale with large experts.
5. **Shared/unique decomposition**: Dead for nonlinear groups. Weight-space
   decomposition is exact, but ReLU applied separately to shared and unique
   groups loses information (+5.7% vs joint, worse than concatenation at -0.2%).
   54% of fine-tuning knowledge is shared. Concatenation remains optimal.

## What Remains

1. ~~Self-routing validation~~: A matrices don't discriminate (Exp 1, DONE).
2. ~~Contrastive routing~~: KILLED at micro scale — domains indistinguishable
   at d=64 (Exp 1b, DONE). Deferred to macro validation with real domains.
3. ~~Sparse routing~~: KILLED at micro scale — top-1 catastrophic (+200% vs
   top-2), phase transition not gradual. k=2 validated as optimal. Deferred
   to macro scale (capacity-bound, 8K params/group insufficient). (Exp 2, DONE)
4. ~~Procrustes decomposition~~: KILLED — nonlinearity breaks weight-space
   decomposition (+5.7% vs joint, worse than concatenation). 54% shared
   knowledge found but can't be cleanly separated in nonlinear capsules.
   May work for linear components (LoRA). (Exp 3, DONE)
5. ~~Scale to N experts~~: Composition scales to N=5 with +1.6% degradation.
   Orthogonality degrades gracefully (cos 0.000→0.112, well under 0.5 concern).
   Calibration scales linearly (200 steps for N=5). **Micro arena exhausted.**
   (Exp 4, DONE)
6. **Beat 1.5B**: The ultimate claim — 0.5B + experts > 1.5B monolithic (Exp 5).
   This is the macro-scale transition. All micro-validated mechanisms carry forward:
   softmax routing, k=2 minimum sparsity, concatenation composition, shared attention.

## Architecture: Expert Library with Softmax Routing (Validated)

```
Input tokens
     │
     ▼
Base model W₀ (frozen)
     │
     ▼
Per layer:
  Expert Library: [group_1, group_2, ..., group_N]
  Each group_i = capsule pool (rank-1 capsules)

  1. Score:   s = x @ W_r^T          (softmax router, trained on reconstruction loss)
  2. Select:  top-k groups by score
  3. Apply:   δ = Σ_selected  w_i · group_i(x)
  4. Output:  base(x) + scale · δ
```

Key properties:
- Softmax router routes by task quality, not domain identity (+0.2% vs joint).
- Shared attention is the composition bottleneck (independent comp fails +13.5%).
- Calibration: ~100 steps on mixed-domain data (scales linearly with N).
- Each group activates at full strength. No dilution.
- k=2 is optimal sparsity at micro scale (k=1 catastrophic, k=2/4/8 within 1.6%).
- Minimum "routing bandwidth" of k=2 required — soft averaging smooths routing uncertainty.
- **Scales to N=5**: +1.6% vs joint at N=5 (vs -0.2% at N=2). Subspaces remain
  orthogonal (cos=0.112 mean, 0.167 max). Protocol validated across domain counts.

## Architecture: Contrastive Expert Library (Deferred to Macro)

```
Per layer:
  Each expert_i = (A_i, B_i, K_i)    -- K_i routing key
  1. Score:   s_i = ||x @ K_i||²     (routing via contrastive key)
  ...
```

Contrastive keys require domain-discriminative hidden states. Validated at
micro scale: domains (a-m vs n-z at d=64) are indistinguishable, so keys
converge to near-random. Re-evaluate at macro scale with distinct domains
(Python vs JavaScript) where BPE creates domain-specific tokens.
