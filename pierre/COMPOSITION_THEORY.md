# Composition Theory: Why Weight-Space Composition Fails on Ternary Models

## The Fundamental Incompatibility

Fine-tuning works via microscopic weight nudges (δ ≈ 0.002, DARE paper).
Ternary weights {-1, 0, +1} force minimum delta of ±1.0.
That's **500x too coarse** for the fine-tuning geometry.

This is why ALL weight-space merge approaches failed:
- v4: continuous delta + re-quantize → signal destroyed (PPL +475%)
- v5.1: LoTA-QAF integer merge → boundary saturation (60M clips)
- v5.2: Bankai row flips → same boundary problem
- Masked Ternary Fine-Tuning → DARE rescaling (×100) impossible in ternary

**Proven theorem:** For a ternary base with K=3 levels, any additive weight
composition that requires DARE-style rescaling is mathematically impossible
because the rescaling factor (1/(1-p)) is a continuous multiplier that
cannot be represented in {-1, 0, +1}.

## The Three Candidates (evaluated)

### 1. Shared Latent Space (continuous bottleneck → quantize)
**FAILS.** Quantization doesn't commute with addition:
Q(W₁ + W₂) ≠ Q(W₁) + Q(W₂)
Delays the interference to quantization step, doesn't prevent it.

### 2. Modular Networks (layer-wise ownership)
**Works but doesn't scale.** Disjoint layers = zero interference.
But tokens must traverse ALL layers, and activation drift accumulates.
N domains × M layers each = impractical depth.

### 3. MoE (conditional computation)
**Works structurally.** Experts isolated in memory — no weight arithmetic.
y = Σ G(x)ᵢ · Eᵢ(x)
But training the router requires non-differentiable gradient estimation.
SMEAR paper shows sparse routing underperforms dense.

## The Two Correct Approaches

### A. Fusion of Experts (output-space composition)
**Merge logits, not weights.**
Y_final = Σ αᵢ · Yᵢ (linear combination of expert outputs)

Each expert's ternary weights are UNTOUCHED.
Composition happens in continuous logit space.
Router determines αᵢ per token.

This is what Pierre SHOULD evolve toward:
- Each domain is a complete ternary model (or ternary adapter)
- At inference: route token to expert, get logits, combine
- No weight merging, no re-quantization, no interference

### B. (IA)³ Activation Scaling
**Scale activations, not weights.**
y = l_expert ⊙ (W · x)

W stays ternary. l_expert is a learned continuous vector (rank-1).
Composition: compose the scaling vectors, not the weights.
l_composed = f(l_1, l_2, ..., l_N)

The scaling vectors live in continuous space where fine-tuning
geometry (δ ≈ 0.002) works perfectly.

## Mathematical Conditions for Correct Composition

For composition to be PROVABLY correct on ternary models:

1. **Never modify ternary weights post-training**
   The weights are the foundation. Composition must happen in a
   continuous space (activations, logits, or scaling vectors).

2. **Expert isolation** (structural, not statistical)
   Either: separate weight matrices (MoE/FoE), or
   separate scaling vectors ((IA)³), or
   separate output streams (logit fusion)

3. **Router must be differentiable** (or closed-form)
   Ridge regression router (our approach) is closed-form — avoids
   the non-differentiable gradient problem of MoE routing.

4. **Composition in continuous space must satisfy Jensen's inequality**
   If experts lie in a convex basin of the logit/activation landscape,
   their average is guaranteed to be at least as good as any individual.
   This requires linear mode connectivity in the ACTIVATION space
   (which is more likely than in weight space for ternary models).

## What This Means for Pierre

Pierre's current approach (Runtime LoRA: y = base(x) + α·(x@A)@B) is
actually closer to the correct answer than we realized:

- The base weights are UNTOUCHED (correct)
- The adapter operates in CONTINUOUS activation space (correct)
- Composition via NRE merge operates on adapter B-matrices (continuous, correct)
- Grassmannian A guarantees orthogonality (correct)

The remaining architectural question: should we move from
"base + adapter side-path" to "multiple complete experts + logit fusion"?

The answer depends on whether the adapter's rank-16 bottleneck is the
quality ceiling, or whether training recipe is the ceiling.
Finding: MiniMax M2.5 proves training recipe > architecture.
Implication: invest in adapter training quality, not architecture changes.
