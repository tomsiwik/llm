# REVIEW-adversarial.md — T4.5: Pierre Adapter Format Compatibility

**Verdict: PROCEED**

## Summary

The core claim — that a lossless bijection exists between MLX and HF PEFT LoRA formats — is mathematically solid. Double-transpose identity is trivial and correct. All 4 kill criteria reported PASS. Finding #433 status of SUPPORTED is appropriate for a structural format verification experiment.

## Blocking Issues

None.

## Non-Blocking Caveats

### 1. K1088 Validation is Weak (PEFT library not installed)
`results.json` shows `"peft_library_available": false`. The actual validation was manual JSON field-level checking, not `peft.LoraConfig.from_pretrained()`. MATH.md offered this as an alternative ("alternatively: manual JSON schema validation passes"), so technically the criterion was met — but the stronger test was not run. For T5 or production, install `peft` and run actual API validation.

### 2. Grassmannian Drift: Prediction-vs-Measurement Discrepancy Not in Table
MATH.md quantitative predictions table says `"verified_max_deviation": "<1e-6"`. Actual measured value is **0.579**. The PAPER.md's "Key Observation" section explains this correctly (training drifts adapters off the initial Grassmannian subspace), but the prediction-vs-measurement table omits this discrepancy. Future experiments should include it explicitly so the training-induced drift is surfaced as a tracked deviation.

**Implication:** T3.x interference proofs used *synthetic* Grassmannian adapters. Real trained adapters have max_cos=0.596 vs synthetic 0.078 (7.6× higher). This gap remains unexplained by a theorem. The routing requirement from T3.1 stands, but the theoretical bound is loose for trained adapters. Not blocking for format compat, but blocking for any future claim about trained-adapter interference bounds.

### 3. Runtime Tests Omitted
K1089 and K1090 are structural format checks only. This is reasonable (no CUDA on M5 Pro), but the experiment would be stronger with a non-CUDA runtime test — e.g., loading with `peft.PeftModel.from_pretrained` on a CPU-only model without CUDA. Not blocking given stated constraints.

## Scientific Integrity

- Prediction-vs-measurement table: present ✓
- Kill criteria consistent with results.json: ✓
- SUPPORTED status appropriate: ✓ (structural verification, not full runtime)
- Grassmannian drift observation: correctly disclosed ✓

## Forward Impact

This finding unblocks T5 user-local training. The Grassmannian drift finding (0.579 deviation after training) is important context for T5 design: if T5 trains adapters, they will drift from the Grassmannian initialization, and any interference guarantee relying on that orthogonality must account for the trained state, not the initialization state.
