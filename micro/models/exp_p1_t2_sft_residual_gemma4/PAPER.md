# PAPER.md — T2.5: SFT-Residual M2P on Gemma 4

**Status: KILLED** | K1044 FAIL

## Abstract

We tested whether zero-initializing ΔB (B_applied = B_sft + ΔB, ΔB_init = 0) is sufficient to
maintain SFT quality during continued adaptation on Gemma 4 E4B. The zero-init guarantee holds
at step 0 (K1046 PASS, acc_step0 = 80%), but 500 steps of gradient descent on the same task data
causes ΔB to grow to 24.5% of B_sft's norm, reducing accuracy from 80% to 58%. K1044 FAILS:
quality_ratio = 0.707 < threshold 0.738. Finding #403 (Qwen3-4B, quality_ratio = 1.175) does NOT
replicate on Gemma 4 with the current training setup.

## Prediction vs Measurement

| Prediction (MATH.md) | Measured | Pass? |
|---|---|---|
| K1046: \|\|ΔB\|\|_F = 0 at step 0 (all 42 layers) | max_diff = 0.0 | ✓ PASS |
| K1045: B_applied time < 10ms | 0.385ms | ✓ PASS |
| acc_step0 = 82% (SFT quality, Theorem 1 Corollary) | 80.0% | ≈ (noise in 50-sample eval) |
| K1044: acc_final ≥ 73.8% (quality_ratio ≥ 0.90) | 58.0% (QR = 0.707) | ✗ FAIL |

## Results

| Metric | Value |
|---|---|
| acc_step0 | 80.0% |
| acc_final | 58.0% |
| quality_ratio | 0.707 |
| K1044 threshold | 73.8% |
| ΔB mean Frobenius | 0.2154 |
| B_sft mean Frobenius | 0.8771 |
| relative_correction | 0.2456 (24.6% of B_sft) |
| training final loss | 1.3594 (from 2.7656) |
| K1045 latency | 0.385ms ✓ |
| K1046 zero-init | verified ✓ |

## Why It Failed

**The zero-init guarantee is a static property (step 0 only), not a dynamic one.**

The gradient at each step is:
```
∂L/∂ΔB_l = (scale × A_l x^T)^T ⊗ (∂L/∂z_l)
```
This is identical to `∂L/∂B_applied_l` — the gradient does NOT know that B_applied = B_sft + ΔB.
Gradient descent moves ΔB in the same direction it would move B, regardless of initialization.

After 500 steps with LR=5e-6 and GRAD_CLIP=0.5, ΔB grew to 24.6% of B_sft norm. This is
sufficient to corrupt the chain-of-thought structure in B_sft (the math reasoning patterns), even
if training NLL decreased (2.77 → 1.36). Training loss measures token prediction on the training
set; eval accuracy measures structured reasoning with exact answer extraction.

**Key distinction from Finding #403 (Qwen3-4B):**
Finding #403 used a non-zero output_scale warmup schedule and trained on *different* data
(personalization queries, not the original SFT task data). This experiment re-trained on the exact
same GSM8K distribution as T2.1 — equivalent to continued SFT with fresh optimizer state, which
is known to cause catastrophic forgetting (Kirkpatrick et al., EWC, arXiv:1612.00796).

## Impossibility Structure

Zero-init of ΔB provides a **type-safe start** (quality at step 0 = SFT quality) but does NOT
prevent forgetting because:

1. `∂L/∂ΔB = ∂L/∂B_applied` — gradients are structurally identical to continued SFT gradients
2. Training on the SAME data distribution with fresh optimizer state overshoots the SFT minimum
3. No regularization term penalizes ||ΔB||_F growth — EWC or L2 penalty on ΔB is required

**Structural guarantee needed (not tried):**
```
L_total = L_task + λ × ||ΔB||_F²   (EWC or elastic anchor)
```
This would bound ||ΔB||_F ≤ O(1/√λ), preventing corruption of B_sft structure.
Alternatively: train ΔB on DIFFERENT data than SFT (the M2P intent).

## Implications

1. **SFT-residual M2P requires data separation**: ΔB must adapt on *new context data*, not
   re-train on the original SFT data. The experiment was testing the wrong scenario.

2. **For personalization (M2P)**: user-specific queries ≠ GSM8K — real M2P data would be
   different enough that the gradient conflict does not occur.

3. **EWC anchor is necessary for same-domain adaptation**: if the use case requires adapting
   on the same domain, add L2 regularization on ΔB with λ tuned to bound relative_correction ≤ 5%.

4. **Finding #403 remains valid**: it used different data (personalization queries) for ΔB.
   T2.5 refutes only the "same-domain re-training via ΔB" variant.

## References

- He et al. (2016, arXiv:1512.03385) — Residual learning
- Kirkpatrick et al. (2017, arXiv:1612.00796) — EWC: Overcoming catastrophic forgetting
- Hu et al. (2022, arXiv:2106.09685) — LoRA
- Finding #403 — SFT-Residual M2P on Qwen3-4B (quality_ratio=1.175, different data)
