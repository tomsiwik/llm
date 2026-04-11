# Current Direction: P4.C1 — Output-Projection SOAP Adapter (v_proj+o_proj)

## Status
All P1 + P2 experiments complete. P3 experiments registered and active.
P3.B0-B4 complete (all killed: domain suppresses personal in all additive strategies).
P3.B5 is the next hypothesis: retrain personal adapter ON domain-adapted model.

## Last Completed: exp_p3_b4_sequential_activation_compose (KILLED — Finding #465)
- Pure additive (24%) WORSE than B-GS (60%) — domain suppresses personal in overlapping directions
- n_overlap_layers=16, power_ratio=1.077 (correct)
- Impossibility: training distribution mismatch — f_P trained on h_base but receives h_base+ΔW_D at inference
- Next: P3.B5 retrain personal adapter ON domain-adapted model

## Active Experiment: exp_p3_b0_medical_oe_adapter (P3.B0)
**Running**: task 0 (full run, ~45-90 min)
**Purpose**: Format-register aligned medical adapter via medalpaca wikidoc (OE explanatory format)
**Root cause**: Finding #457 killed MCQ training due to format-register mismatch. Fix: train on open-ended explanatory text matching the evaluation register.
**Kill criteria**:
- K1169: improvement_rate (vocabulary rubric) > 80% (vs 60% MCQ, Finding #457)
- K1170: adapted_mean_vocab >= 1.5 × base_mean_vocab
- K1171: MMLU regression < 5pp

## Queued Experiments
1. **exp_p3_b1_ortho_t2t3_compose** (priority 2) — Gram-Schmidt re-orthogonalization fixes T2+T3 simultaneous composition
   - Structural fix for Finding #460 (ε_B=0.1607 > 0.1, power_ratio=2.96x)
   - Algebraic verification, fast (<10min)

2. **exp_p3_c0_pipeline_behavioral_e2e** (priority 3) — Full pipeline E2E
   - Depends on P3.B0 and P3.B1

## P2 Tier (both killed)
- exp_p2_a0_medical_pubmedqa_adapter: KILLED (base near-random, δ=+0.015)
- exp_p2_a1_tier2_plus_tier3_composition: KILLED (ε_B × power = 0.476 >> 0.132)

## P1 Tier (all complete: 40+ supported, killed findings)
Key results:
- N=100 Grassmannian composition: max_cos=2.25e-8 (Finding #440)
- Ridge routing N=25: 98.8% (Finding #458)
- Personal adapter: 76pp compliance gain (Finding #436)
- Dynamic registry: all ops O(1)/fast (Finding #454)
- Flywheel viable: 3-promotion ε_cumul=7.6% (Finding #453)

## Key Insight from KILLED Experiments
1. Domain adapters (MCQ format) → δ_D ≈ 0 for Gemma 4 (Finding #457)
2. Medical shows highest gap: base_mean=1.4, MCQ adapted=2.1 (60% rate)
3. OE training hypothesis: format-register alignment → δ_D > 0

## When P3.B0 Completes
- If SUPPORTED: proceed to P3.B1 (composition fix) then P3.C0 (full pipeline E2E)
- If KILLED: derive impossibility structure, pivot to subdomain-specific adapters
  or conclude δ_D ≈ 0 for capable base models regardless of format

## Updated: P4.B0 — Domain Adapter Quality Benchmark (KILLED — Finding #477)
- Math +20pp, Finance +14.7pp, Code +6.7pp, Legal +9.3pp, Medical -4pp
- K1224 FAIL: 2/5 < 3/5 (borderline: legal 9.3pp just below 10pp threshold)
- K1225 FAIL: retention 0.890 < 0.90 (math adapter worst: 0.834)
- K1226 FAIL: avg adapted acc 0.480 < 0.50
- Root cause: rank-6 adapters can't overcome strong base priors (medical/legal base ~45-48%)
- Next: P4.B1 (harder domain questions or rank-16 adapters for medical/legal)

## Updated: P4.A2 — 6-Domain System Integration

### Last Completed: exp_p4_a1_domain_adapter_speedtest (SUPPORTED — Finding #475)
- "New domain in <10 min" verified: 7.53 min total, +20pp behavioral, 7.61 MB adapter

### Active Experiment: exp_p4_a2_6domain_integration (P4.A2)
**Running**: task 0 (full run, ~10-20 min)
**Purpose**: Extend 5-domain router to 6 domains (add biology) and test full pipeline
**Design**: 6-domain TF-IDF ridge, reuse biology adapter from P4.A1
**Smoke**: ALL_PASS (98.3% acc, 12.5ms train, +33pp, 100% bio precision)
**Key finding**: cos(biology, medical) = 0.062 — biology is geometrically isolated!
**Kill criteria**:
- K1220: 6-domain routing accuracy >= 93%
- K1221: router re-train time < 1 second
- K1222: biology adapter improvement >= 10pp
- K1223: biology routing precision >= 85%

## Updated: P4.A1 — Domain Adapter Speedtest

### Last Completed: exp_p4_a0_5domain_routing (SUPPORTED — Finding #474)
- 5-domain TF-IDF Ridge: 97.3% weighted accuracy, 0.247ms p99

### Active Experiment: exp_p4_a1_domain_adapter_speedtest (P4.A1)
**Running**: task 0 (full run, ~5-8 min)
**Purpose**: Verify vision claim "new domain in <10 min" — train biology adapter from scratch
**Design**: rank=16, 200 steps, 12 LoRA layers (q_proj), biology domain (6th domain)
**Smoke**: ALL_PASS (training=0.15min, improvement=40pp, size=7.61MB)
**Kill criteria**:
- K1217: training_time < 10 min
- K1218: behavioral_improvement ≥ 10pp (vocabulary rubric, ≥8 bio terms threshold)
- K1219: adapter_size < 10 MB

## Updated: P4.B1 — Gap-Targeted Evaluation (KILLED — Finding #478)

### Key Discovery: Notation Artifact
- P4.B0 math +20pp was NOT a knowledge gap — it was notation alignment
  - P4.B0 keywords: "a^2", "u dv", "f(g(x))" — notation NOT produced in prose
  - P4.B1 keywords: "Zorn", "maximal element" — natural vocabulary produced by Gemma 4
- Gemma 4 4B has no exploitable knowledge gap in standard academic domains
- Base scores on advanced questions: math=0.633, medical=0.600, legal=0.433, code=0.467, finance=0.433

### Structural Impossibility
δ_d > 0 requires BOTH:
1. H(V_d|θ_base) > H_threshold (vocabulary gap)
2. V_d ∩ V_train ≠ ∅ (adapter training covers question vocab)
Gemma 4 4B fails (1) for academic domains. P1 T2 adapters fail (2) for advanced questions.

### Next: P4.C — Formatting gaps (not knowledge gaps)
- LaTeX notation output for math responses
- Clinical note format (SOAP, ICD-10 coding)
- Legal document structure (pleadings, motions)
- Code in specific framework style (FastAPI, Rust async)
