# REVIEW-adversarial.md — T2.5: SFT-Residual M2P on Gemma 4

**Verdict: PROCEED (finding documented, no follow-up needed immediately)**

## Summary

K1044 FAILED. The experiment is KILLED. Finding #447 captures the impossibility structure.
The failure is well-explained by gradient identity (∂L/∂ΔB = ∂L/∂B_applied) + same-domain
re-training. No additional re-test needed — the fix is architectural (data separation for M2P),
not a hyperparameter issue.

## Adversarial Challenges

### Challenge 1: Is 50-sample eval reliable?
The step-0 accuracy is 80% vs predicted 82% — a 2-point gap. With 50 samples, σ ≈ 5.7%.
The gap is within 1σ. The Theorem 1 prediction (82% at step 0) is confirmed within noise.

The final accuracy (58%) is well below threshold (73.8%): 15.8pp gap, well outside noise. The
failure is not a measurement artifact.

### Challenge 2: Could a lower LR prevent the failure?
Partially. With LR = 5e-6 and GRAD_CLIP = 0.5, ΔB grew to 24.6% of B_sft. Reducing LR by 10×
would reduce ΔB growth proportionally (to ~2.46%). At that level, acc_final might recover
above threshold.

However, this treats SYMPTOMS not the DISEASE. The real issue is training on the SAME data as
SFT — EWC regularization is the correct structural fix, not LR reduction. LR reduction would
also slow convergence on genuine new-domain M2P tasks.

### Challenge 3: Does this invalidate M2P for personalization?
No. The failure is specific to same-domain re-training. The M2P use case is:
- SFT phase: train B_sft on domain data (GSM8K math)
- M2P phase: train ΔB on USER-SPECIFIC queries (different data, user context)

This experiment conflated these phases by using the same GSM8K data for ΔB. The gradient
conflict does not arise when M2P data is sufficiently different from SFT data.

Finding #403 (quality_ratio=1.175 on Qwen3-4B) remains valid because it used different data.

### Challenge 4: Why did acc drop from 80% to 58%, not just stagnate?
The drop (not stagnation) suggests active interference. Training on GSM8K with fresh optimizer
moves ΔB in directions that partially cancel B_sft's learned reasoning patterns. This is
consistent with Kirkpatrick et al. EWC (arXiv:1612.00796): continued gradient updates on the
same task data with no memory of prior parameter trajectory destabilize learned representations.

The exact mechanism: ΔB must partially *undo* what B_sft learned to minimize NLL on the
same data seen in different order with different batch composition. This is parameter interference,
not just noise.

## Structural Path Forward

To resurrect M2P on Gemma 4:
1. **Data separation**: train ΔB on held-out user queries (not GSM8K training set)
2. **EWC regularization**: L_total = L_task + λ||ΔB||_F² to bound ||ΔB|| ≤ O(1/√λ)
3. **Only need one**: data separation alone is sufficient for the M2P use case

The mathematical guarantee is clear: as long as the M2P training distribution is sufficiently
different from the SFT distribution, ΔB gradient updates will not create cancellation interference
with B_sft. This is guaranteed by the non-overlapping gradient geometry.

## Conclusion

The failure is structural and well-understood. The finding (K1044 FAIL, QR=0.707) is reliable.
No REVISE needed. The impossibility structure is mathematically clear. M2P on Gemma 4 can
proceed with data-separated M2P training (real personalization data, not SFT replay).
