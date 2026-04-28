# Learnings — N=3 Routing Accuracy

## Key Finding

Domain routing is trivially solvable (100% with TF-IDF + Ridge) but single-adapter
routing underperforms uniform composition by 3.3pp. The adapters are complementary,
not interfering.

## What This Means for Pierre

1. **Hard routing is suboptimal.** Selecting one adapter per sample throws away
   beneficial cross-domain contributions.

2. **Soft routing is the next hypothesis.** Router should output weights
   (e.g., [0.7, 0.2, 0.1]) not argmax. This preserves complementary benefits
   while emphasizing the primary domain.

3. **The "interference" framing was wrong.** F#825 + this experiment together show
   that uniform and routed are both suboptimal — the ideal is somewhere between
   (weighted combination tuned to the sample).

## Reusable Patterns

- TF-IDF + Ridge is a strong baseline for domain classification (100% on 3 domains)
- ComposedLinear wrapper is mandatory for quantized models
- mlx-lm LoRA convention: A=(d_in,r), B=(r,d_out), delta = A @ B
