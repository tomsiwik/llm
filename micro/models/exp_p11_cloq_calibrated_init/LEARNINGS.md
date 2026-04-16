# LEARNINGS.md — P11.K0: CLoQ Calibrated LoRA Initialization

**Status**: Pre-run context notes (experiment queued as pueue task 26 after bug fix)

## Core Idea
CLoQ (arXiv:2501.18475) initializes LoRA weights to exactly compensate quantization error
(W_float - W_4) via Eckart-Young optimal low-rank approximation. Start at the right place
instead of random zero — adapter capacity goes to reasoning, not artifact repair.

## Math Validity
- Theorem algebraically correct, verified by reviewer
- Eckart-Young is classical (1936), proof is tight
- Adapter key format confirmed matches mlx_lm s1K output — no silent load failure
- 8-bit proxy captures ≥15/16 of true quantization error (ratio bound 1/16)

## Key Predictions to Verify in PAPER.md
1. Top-8 SVs capture ≥70% of ||E||_F² energy (per-group quant structure)
2. CLoQ calibration < 10 min (K1536)
3. CLoQ-init training ≥ s1K + 2pp MMLU-Pro (K1535) — baseline TBD pending s1K result
4. CLoQ + s1K data → ≥66% MMLU-Pro with thinking (K1537)

## Primary Failure Mode
Gemma 4's 4-bit model may already be near-float quality for reasoning tasks.
If ||E||_F is small, CLoQ correction is negligible → K1535 fails (+2pp won't appear).
K1536 (calibration speed) and K1537 (absolute accuracy) may still pass.

## PAPER.md Must Include
- Actual s1K baseline (pending task 0) — not 65% placeholder
- Hyperparameter parity confirmation (rank=8, scale=1.0, lr=1e-5, 1000 steps)
- Measured SVD energy capture per layer type (v_proj, o_proj)
- delta_vs_s1K table (the core comparison)

## Bug Fix Applied (2026-04-14)
MLX 4-bit quantized weights dequantize to **bfloat16**, not float32. NumPy's PEP 3118 buffer
protocol rejects bfloat16 (item size 2 maps to 'B' format → `RuntimeError: Item size 2 for
PEP 3118 buffer B`). Fix: add `.astype(mx.float32)` before `np.array()` in both
`dequantize_layer_weight()` and `get_raw_linear_weight()`. The cast is lossless (bfloat16 →
float32 expands mantissa); SVD proceeds in float32 as designed. Task requeued as pueue task 26.

## Implications for Next Experiment
If CLoQ works (+2pp): initialization quality matters more than architecture tweaks;
apply CLoQ init to all future adapter training.
If CLoQ fails (||E|| too small): Gemma 4's 4-bit quantization is already high-fidelity;
focus on data quality and RL refinement (GRPO) rather than quantization compensation.
