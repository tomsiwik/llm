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

## The Fix: Decoupled Contrastive Routing Keys

Add a thin **routing key** `K_i` per expert, trained contrastively at registration time:

- **K_i (d_in, d_key)**: routing discriminator — trained with InfoNCE loss
- **A_i, B_i**: computation — frozen, unchanged from original training

Expert quality preserved. Routing keys are calibrated on ~50 samples/domain
in ~50 optimization steps (~seconds). Adding expert N+1 only requires
recalibrating routing keys.

## What Remains

1. ~~Self-routing validation~~: A matrices don't discriminate (Exp 1, DONE).
2. **Contrastive routing**: Calibrate K_i keys, target >85% accuracy (Exp 1b).
3. **Sparse routing**: Show top-1 expert selection matches full CAT quality
   at 1/N compute (Exp 2).
4. **Procrustes decomposition**: Separate shared/unique, eliminate dilution,
   route the residuals (Exp 3).
5. **Scale to N experts**: 5+ languages, verify subspaces stay orthogonal (Exp 4).
6. **Beat 1.5B**: The ultimate claim — 0.5B + experts > 1.5B monolithic (Exp 5).

## Architecture: Contrastive Expert Library

```
Input tokens
     │
     ▼
Base model W₀ (frozen)
     │
     ▼
Per layer:
  Expert Library: [expert_1, expert_2, ..., expert_N]
  Each expert_i = (A_i, B_i, K_i)

  1. Score:   s_i = ||x @ K_i||²      (routing via K, not A)
  2. Select:  top-k experts by score
  3. Apply:   δ = Σ_selected  w_i · (x @ A_i) @ B_i   (computation via A@B)
  4. Output:  base(x) + scale · δ
```

Key properties:
- Adding expert N+1 = store (A, B), recalibrate K. No retraining of experts.
- Routing cost = N × d_key dot products per token. Negligible for N < 100.
- Each expert activates at full strength. No dilution.
- Routing keys are ~25% parameter overhead (d_key=8 vs rank=16).
- Experts can be hot-swapped, versioned, A/B tested.
