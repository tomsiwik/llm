# LEARNINGS.md — T4.5: Pierre Adapter Format Compatibility

## Core Finding
Pierre (MLX) adapters convert losslessly to HF PEFT/vLLM/Unsloth format via a double-transpose bijection. Format lock-in risk is zero.

## Why
The MLX weight convention stores lora_a as [d_in, r] and lora_b as [r, d_out]; PEFT expects the transposes. A simple key rename + transpose is an exact involution — double-applying it recovers the original, with round-trip error 0.0.

## Critical Caveat: Grassmannian Drift After Training
Trained adapter A-matrices have max deviation from A^T·A ≈ I_r of **0.579** — far from the initialization value of ~0 (by construction). This means:
1. T3.x interference proofs (which assumed Grassmannian structure) are valid only for synthetic or freshly-initialized adapters, not for trained adapters.
2. Real adapters show max_cos=0.596 at N=5 vs synthetic 0.078 — the gap is training-induced rotation, not numerical noise.
3. Any future interference bound for trained adapters requires a new theorem (e.g., bounding cosine under SGD drift, or re-orthogonalizing A matrices post-training via QR).

## Implications for Next Experiment
T5 (user-local training) can safely produce PEFT-compatible adapters using the bijection. However, T5 must account for trained-adapter interference bounds being looser than the Grassmannian predictions — either re-orthogonalize A after training or derive a bound for the drifted case.
