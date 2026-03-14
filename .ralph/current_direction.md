# Current Direction (2026-03-14)

## Just Integrated: Two macro experiments PROVEN

### 1. exp_cat_weight_convergence (P3, macro) -- PROVEN
**Node:** exp_cat_weight_convergence
**Title:** Do CAT-optimized weights converge to ~1.0 with orthogonal specialized experts?
**Dir:** macro/cat_weight_convergence
**Deps:** exp_oae_vs_lora_soups (proven), exp_distillation_pilot_50 (supported)
**Status:** PROVEN

**Results:**
- K1 SURVIVES: CAT weights converge near 1.0 (mean|w-1| within 0.1 threshold)
- K2 SURVIVES: PPL improvement from CAT ≤5% over unit weights
- Validates SOLE unit-weight assumption: orthogonal experts make CAT unnecessary
- Base PPL 7.1748, Qwen2.5-7B, 50 pilot adapters, rank-16

### 2. exp_attention_layer_orthogonality (P2, macro) -- PROVEN
**Node:** exp_attention_layer_orthogonality
**Title:** Attention-layer LoRA adapters maintain structural orthogonality for dissimilar domains at macro scale
**Dir:** macro/attention_layer_orthogonality
**Deps:** exp_structural_orthogonality_proof (proven)
**Status:** PROVEN

**Results:**
- K1 PASS: 0.0% of dissimilar pairs exceed sqrt(r/d) bound (threshold 20%)
- K2 PASS: max attention cos = 0.0 for dissimilar domains (threshold 0.1)
- Bound sqrt(16/3584) = 0.0668 at d=3584, rank=16
- Structural orthogonality extends to attention layers, not just MLP

## GPU Queue Status
- ACTIVE: pilot50_held_out_eval running
- 12 PENDING tasks queued (composition quality, orthogonality, distillation quality, clone-compete, reasoning expert)
- Prior failures requeued: composition_quality (IndexError), measure_orthogonality

## Context
- Strong SOLE theory validation: orthogonality + unit weights confirmed at macro scale
- Multiple active experiments across phases 1-3 being processed in GPU queue
