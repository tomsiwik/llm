# LGME Ablation Findings

## 32-Combination Ablation Study

All 2^5 combinations of phase flags were tested:

| Phase | Flag | Effect on Loss | Verdict |
|-------|------|----------------|---------|
| 1. ART-modulated LR | `PHASE1_ART_LR` | +0.028 (hurts) | Starves optimizer on KNOWN inputs |
| 2. Bloom filter gate | `PHASE2_BLOOM_GATE` | ~0.000 (none) | Zero effect on loss |
| 3. Splay cache | `PHASE3_SPLAY_CACHE` | ~0.000 (none) | Zero effect on loss |
| 4. MoE routing | `PHASE4_MLP_ROUTING` | -0.003 (helps) | Only phase that improves loss |
| 5. ART spawn | `PHASE5_ART_SPAWN` | +0.005 (hurts) | Adam buffer disruption on consolidation |

## Key Results

- **ALL ON was the worst configuration** — phases compound each other's harm
- **MoE routing is the only beneficial phase** — slight but consistent improvement
- **Bloom filter, Splay tree, HNSW**: zero measurable effect on training loss
- **ART-LR modulation**: actively harmful — scaling down LR for "known" inputs prevents the optimizer from refining already-learned patterns
- **Spawning/consolidation**: hurts because merging experts disrupts Adam momentum/variance buffers

## Pivot Decision

The "cognitive stack as routing optimizer" narrative is dead. The valuable research direction is **continual learning without full retraining**, where MoE expert isolation is the core mechanism.

### Supporting Literature (ICLR 2025, Feb 2025)

- MoE expert isolation at the FFN layer works for preventing forgetting
- **Shared attention is the forgetting bottleneck** — not the experts
- EWC on shared attention params is the known fix
- Kohonen routing for MoE-CL appears to be novel (no published precedent found)

## Archived Artifacts

- `archive/ablation.py` — the 32-combination ablation runner
- `archive/ablation_results.csv` — raw results
- `archive/ablation_chart.png` — visualization

---

## Contrastive Routing Keys — 2026-03-04

**Status: KILLED** at micro scale. 3 of 4 kill thresholds exceeded.

### Setup
- ContrastiveRouterGPT: CapsuleMoEGPT + InfoNCE-trained routing keys K_i
- d=64, G=4/domain, D=2 (a-m vs n-z), N=8 composed groups, d_key=8
- Protocol: pretrain shared base → fine-tune capsule groups/domain → compose → calibrate keys with InfoNCE on ~128 hidden states/domain, 50 steps
- 3 seeds (42, 123, 7), 219K total params

### Results (3-seed aggregate)

| Metric | Result | Threshold | Verdict |
|--------|--------|-----------|---------|
| Routing accuracy | 53.3% | >85% (kill <70%) | KILLED |
| Composition quality | +141% vs joint | <5% (kill >10%) | KILLED |
| vs linear probe | 53.3% < 59.8% | Must beat | KILLED |
| Sample/step efficiency | 128 samples, 50 steps | <100 each | OK |

### Root Cause
**MATH.md Assumption 6 falsified**: at d=64 with character-level a-m vs n-z
tokenization, hidden states are NOT domain-discriminative. Even a linear probe
only achieves ~60%. The contrastive loss has no signal to learn from.

### Key Insight
**Task-routing > identity-routing** for similar domains. The softmax router
works (+0.2% vs joint) because it optimizes reconstruction loss — routing
tokens to whichever groups minimize prediction error, regardless of domain
identity. Contrastive keys attempt explicit domain discrimination via InfoNCE,
but there's no domain signal at micro scale to learn.

### Implications
- Contrastive routing deferred to macro validation (Python vs JavaScript, d=256+, BPE tokens)
- Softmax router calibration validated as the composition routing baseline
- Sparse routing (Exp 2: top-1 matching top-2 quality) is next — doesn't depend on domain discrimination
- Risk at macro scale: if domains share representation structure, contrastive keys may still underperform reconstruction-based routing

### Artifacts
- `micro/models/contrastive_router/` — code, tests, MATH.md, PAPER.md
- Parent model: `capsule_moe`

---

## Sparse Routing (Top-k Sweep) — 2026-03-04

**Status: KILLED** (top-1). **k=2 VALIDATED** as optimal sparsity.

### Setup
- SparseRouterGPT: CapsuleMoEGPT + runtime top_k control, 0 new params
- d=64, G=8 composed (4/domain), top_k sweep {1, 2, 4, 8}
- Protocol: pretrain shared base → fine-tune capsules/domain → compose → calibrate fresh router per k (100 steps)
- 3 seeds (42, 123, 7), 203K total params

### Results (3-seed aggregate)

| Setting | Val Loss | vs Joint | vs k=2 |
|---------|----------|----------|--------|
| joint | 0.5188 | baseline | — |
| learned k=1 | 1.5799 | +204.5% | +200.6% |
| learned k=2 | 0.5256 | +1.3% | baseline |
| learned k=4 | 0.5184 | -0.1% | -1.4% |
| learned k=8 | 0.5172 | -0.3% | -1.6% |
| uniform k=1 | 3.9545 | +662.2% | — |
| uniform k=2 | 1.1117 | +114.3% | — |

### Router Analysis (G=8)

| k | H(p) | H/H_max | C_1 | Domain% |
|---|------|---------|-----|---------|
| 1 | 1.791 | 0.861 | 0.285 | 50.4% |
| 2 | 1.573 | 0.756 | 0.354 | 50.5% |

### Kill Threshold Checks

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Top-1 vs top-2 degradation | +200.6% | >10% | **KILL** |
| Learned vs uniform at k=1 | Inconsistent (2/3 seeds) | Loses | PASS* |
| Router entropy at k=1 | 0.861 H_max | >0.9 | PASS |
| Top-1 vs joint degradation | +204.5% | >15% | **KILL** |

### Root Cause
**Hard selection amplifies flat probability distribution.** At k=1, w_{g*} = 1.0 regardless of router confidence (C_1 = 0.285, only 29% mass). The 71% probability mass across 7 silenced groups is lost. At k=2+, soft averaging smooths routing uncertainty. This creates a **phase transition** between k=1 and k=2, not gradual degradation.

### Key Insights
1. **Phase transition, not gradual tradeoff**: k=1 catastrophic (+200%), k=2/4/8 within 1.6% of each other
2. **Quality-compute tradeoff flat above k=2**: the "knee" is between k=1 and k=2
3. **Learned routing prevents bad routing** (vs uniform: catastrophic) but doesn't achieve great routing (+1.3% vs joint)
4. **Domain alignment ~50% at all k**: router never learns domain discrimination (consistent with contrastive_router finding)
5. **Capacity-bound, not mechanism-bound**: 8K params/group too few for k=1. Switch Transformer uses k=1 at scale with large experts

### Implications
- Sparse routing (k=1) deferred to macro scale — requires larger capacity per group
- k=2 validated as optimal composition sparsity at micro scale
- Phase transition suggests minimum "routing bandwidth" threshold — formalizable
- Next: Procrustes decomposition (Exp 3) or scale to 5+ experts (Exp 4)

### Artifacts
- `micro/models/sparse_router/` — code, tests, MATH.md, PAPER.md
- Parent model: `capsule_moe` (0 new params, hyperparameter sweep only)

---

## Shared/Unique Decomposition (Procrustes) — 2026-03-04

**Status: KILLED.** Decomposed composition exceeds 5% kill threshold (+5.7% vs joint).

### Setup
- ProcrustesDecompGPT: shared (always-on) + unique (routed) capsule pools
- Decomposition: shared_delta = avg(delta_A, delta_B), unique = delta - shared
- d=64, 4 shared groups + 8 unique groups (4/domain), n_caps=64
- Protocol: pretrain shared base → fine-tune capsule groups/domain → decompose → compose
- 3 seeds (42, 123, 7), 12 groups total (vs 8 for concatenation)

### Decomposition Analysis (3-seed aggregate)
- Shared fraction of delta norm: 53.9% (range: 53.6%-54.1%)
- Reconstruction error: <6e-08 (numerically exact in weight space)

### Results (3-seed aggregate)

| Method | Avg Val Loss | vs Joint |
|--------|-------------|----------|
| Joint training | 0.5225 | baseline |
| Concat + calibrated | 0.5213 | -0.2% |
| Concat + uniform | 0.6248 | +19.6% |
| Task arithmetic | 0.7540 | +44.3% |
| Shared only | 0.6781 | +29.8% |
| Decomp + calibrated | 0.5525 | +5.7% |
| Decomp + uniform | 0.6206 | +18.8% |

### Kill Threshold Checks

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Decomp+cal vs joint | +5.7% | <5% | **KILL** |
| Decomp+cal vs concat+cal | 5.9% worse | Must not be worse | **KILL** |
| Shared fraction | 53.9% | >5% | PASS |
| Decomp+uniform vs concat+uniform | 0.7% better | Must be better | PASS (marginal) |

### Root Cause
**Nonlinearity breaks weight-space decomposition.** The shared/unique split is
exact in weight space (base + shared + unique = base + delta), but NOT in
function space. ReLU applied separately to shared_group(x) and unique_group(x)
differs from ReLU applied to (shared+unique)(x). The unique groups have tiny
weights → many activations killed by ReLU → information loss.

### Key Findings
1. **54% of fine-tuning is shared**: domains learn substantial common knowledge
2. **Task arithmetic dilutes catastrophically**: +44.3% degradation, confirming VISION.md
3. **Shared alone insufficient**: +29.8%, proving unique knowledge is essential
4. **Concatenation remains optimal**: -0.2% vs joint, no decomposition needed
5. **Marginal robustness**: decomp+uniform is 0.7% better than concat+uniform (weak signal)

### Implications
- Weight-space decomposition killed for nonlinear capsule groups
- Concatenation protocol remains the validated composition method
- Decomposition may work for LINEAR expert components (LoRA adapters: ΔW = A@B, no nonlinearity)
- Next: Exp 4 (scale to 5+ experts) — the natural continuation

### Artifacts
- `micro/models/procrustes_decomp/` — code, tests, MATH.md, PAPER.md
- Parent model: `capsule_moe`

---

## N=5 Expert Scaling — 2026-03-04

**Status: PASSED.** Composition protocol scales to N=5 domains.

### Setup
- NExpertScaleGPT: CapsuleMoEGPT with 5 domain expert sets composed
- 5 domains by first letter: a-e (10.5K), f-j (5.0K), k-o (8.6K), p-t (5.6K), u-z (2.4K)
- d=64, G=20 composed (4/domain), k=10 (2/domain), 64 caps/group
- Protocol: pretrain shared base (300 steps) → fine-tune capsules/domain
  (300 steps × 5) → compose by concatenation → calibrate router (200 steps)
- 3 seeds (42, 123, 7), ~732K total params

### Results (3-seed aggregate)

| Method | Avg Val Loss | vs Joint |
|--------|-------------|----------|
| Joint training (G=20, k=10) | 0.4951 | baseline |
| Composed + calibrated | 0.5032 | +1.6% |
| Composed + uniform | 0.6910 | +39.6% |
| Task arithmetic | 0.5228 | +5.6% |

### Per-Domain Breakdown

| Domain | Size | vs Joint |
|--------|------|----------|
| a_e | 10,479 | -0.1% |
| f_j | 4,973 | +3.0% |
| k_o | 8,613 | -0.2% |
| p_t | 5,609 | +2.6% |
| u_z | 2,359 | +3.0% |

### Subspace Orthogonality

| Metric | N=5 | N=2 (prior) |
|--------|-----|-------------|
| Mean cosine sim | 0.112 | ~0.000 |
| Max cosine sim | 0.167 | ~0.000 |

Orthogonality degrades gracefully: well below the 0.5 concern threshold.

### Kill Threshold Checks

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Composition+cal vs joint | +1.6% | <5% | PASS |
| Max pairwise cosine sim | 0.167 | <0.5 | PASS |
| Any single domain >10% | max +3.0% | <10% | PASS |
| Calibration scaling | 200 steps OK | <400 | PASS |

### Key Findings
1. **Composition degrades sub-linearly**: +1.6% at N=5 vs -0.2% at N=2
2. **Orthogonality scales**: cos 0.000→0.112, suggesting ~0.5 concern around N≈9-10
3. **Calibration scales linearly**: 200 steps for N=5 (2× for 2.5× domains)
4. **Smaller domains degrade more**: data quantity per domain affects quality
5. **Router remains near-uniform**: H/H_max ≈ 0.999, no domain specialization
6. **Task arithmetic still fails**: +5.6% (less catastrophic than N=2's +44% due to smaller per-domain scale)

### Implications
- **Micro arena exhausted**: 5 experiments have systematically explored composition
  at d=64. Remaining questions are scale-bound, not mechanism-bound.
- The validated micro protocol: pretrain base → fine-tune capsules/domain →
  concatenate → calibrate softmax router → k=2 minimum sparsity
- **Next: Exp 5 (macro scale)** — transition to 0.5B real LLM + LoRA experts,
  beat 1.5B monolithic model

### Artifacts
- `micro/models/n_expert_scale/` — code, tests, MATH.md, PAPER.md, REVIEW-adversarial.md
- Parent model: `capsule_moe`
