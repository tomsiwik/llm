# LEARNINGS.md — T2.5: SFT-Residual M2P on Gemma 4

**Status: KILLED** | Finding #447

## Core Finding

Zero-init of ΔB (B_applied = B_sft + ΔB, ΔB_init=0) guarantees SFT quality only at step 0.
Training on the same domain data for 500 steps degraded accuracy from 80% to 58% (QR=0.707 < 0.738 threshold).

## Why

The gradient identity ∂L/∂ΔB = ∂L/∂B_applied means gradient descent moves ΔB identically to
how it would move B — initialization provides no protection against catastrophic forgetting.
Training on the same GSM8K data as SFT with fresh optimizer state is structurally equivalent to
continued SFT; ΔB grew to 24.6% of B_sft's Frobenius norm, partially canceling learned reasoning
patterns (Kirkpatrick et al., EWC, arXiv:1612.00796).

## Implications for Next Experiment

M2P on Gemma 4 requires **data separation**: ΔB must be trained on new-context user queries,
not the original SFT distribution. Finding #403 (quality_ratio=1.175, Qwen3-4B, different data)
remains valid — this experiment tested the wrong scenario. EWC regularization
(L_total = L_task + λ||ΔB||_F²) is the structural fix for same-domain adaptation if needed.

## Key Numbers

| Metric | Value |
|---|---|
| acc_step0 | 80% (expected 82%) |
| acc_final | 58% |
| quality_ratio | 0.707 (threshold 0.738) |
| relative_correction | 24.6% of B_sft |

## References

- Kirkpatrick et al. (2017, arXiv:1612.00796) — EWC: catastrophic forgetting
- Finding #403 — SFT-Residual M2P Qwen3-4B (different data, QR=1.175, SUPPORTED)
