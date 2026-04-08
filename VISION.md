# Vision: Composable Domain Experts via M2P Distillation

## Core Thesis

A language model is not a monolith — it is a frozen base plus composable domain experts.
Each expert is generated from context in one forward pass, with zero parameter-space
interference by construction, and composable without retraining.

**Platform:** Apple M5 Pro, 48GB unified memory. This IS the deployment target.
**Framework:** MLX only. No CUDA.

## Architecture: Decoupled Guarantees

```
Frozen Base (Qwen3-4B or similar)
    │
    ├── Frozen Grassmannian A-matrices (orthogonal slots, generated once via QR)
    │     → Mathematical guarantee: A_i^T A_j = 0 by construction
    │
    ├── M2P-generated B-matrices (domain content, generated per context)
    │     → Learned: M2P transformer reads hidden states → outputs LoRA B
    │     → 97.7-100.6% of SFT quality at toy scale (Findings #359, #361)
    │     → 1.15ms generation time (Finding #339)
    │
    └── Scale via preservation loss (learned, not fixed)
          → L_preserve = CE(base + adapter, general_tokens)
          → Gradient teaches M2P the correct scale automatically
          → scale=5 gives 0pp MMLU degradation (Finding #330)
```

### Parameter-Space Orthogonality (proven)

For adapters Δ_i = B_i A_i and Δ_j = B_j A_j:

```
⟨Δ_i, Δ_j⟩_F = trace(A_i^T B_i^T B_j A_j)
               = trace(B_j (A_j A_i^T) B_i^T)     [cyclic permutation]
               = trace(B_j × 0 × B_i^T)            [A_j A_i^T = 0]
               = 0
```

**This holds for ANY B_i, B_j.** The Grassmannian A-slots guarantee zero
parameter-space interference. Verified at float32 zero (Finding #341 K848).

### Activation-Space Interference (empirically bounded, NOT guaranteed)

Parameter-space orthogonality does NOT guarantee activation-space orthogonality.
The actual output is: h_out = W_base·x + B_1(A_1·x) + B_2(A_2·x).
Even with orthogonal A-slots, B_1(A_1·x) and B_2(A_2·x) write to the same output
space and can destructively interfere.

**Empirical evidence:** max activation-space |cos| = 0.29 at N=5 (Finding #353).
This is bounded in practice but has no mathematical guarantee. Measuring how this
scales with N is an open experiment (Level 2B in the PoC roadmap).

**Scale:** scale=20 destroys MMLU by -60pp (Finding #320). scale=5 is safe (#330).
Preservation loss teaches M2P to self-calibrate.

## Four Tiers of Knowledge

```
┌─────────────────────────────────────────────────────────┐
│ Tier 4: Session Adapters (ephemeral, per-conversation)  │
│   Generated from context in one M2P forward pass.       │
│   Lives only during the session. ~1KB-10KB.             │
├─────────────────────────────────────────────────────────┤
│ Tier 3: User Adapters (persistent, per-user)            │
│   Distilled from accumulated sessions via M2P.          │
│   Updated after each session. ~100KB.                   │
├─────────────────────────────────────────────────────────┤
│ Tier 2: Domain Adapters (shared, SFT-trained)           │
│   "Medical", "Code", "Legal" — crystallized from users. │
│   Candidates for promotion to base. ~1-10MB.            │
├─────────────────────────────────────────────────────────┤
│ Tier 1: Base (frozen pre-trained, grows via promotion)  │
│   Qwen3-4B or self-grown through promotion cycles.      │
│   Each promotion adds a solidified expert permanently.  │
└─────────────────────────────────────────────────────────┘
```

## What We Proved (427 experiments, 358 findings)

### Composition Guarantees (Conclusive — parameter-space only)

| Finding | Result |
|---------|--------|
| #3 | Grassmannian orthogonality: cos=0.0002 at d=2560 (50x below theory) |
| #126 | Structural orthogonality is geometric guarantee: 17-69x below Welch bound |
| #341 K848 | A-slot orthogonality verified at float32 zero on M2P-generated adapters |
| #334 | Pre-sum without routing = unrouted mixture — routing IS the matmul |

### M2P Quality Scaling (Supported — toy scale, synthetic domains)

| Finding | Result |
|---------|--------|
| #359 | M2P at d=256: 97.6% of SFT with n≥T data scaling |
| #361 | M2P at d=512: 100.6% of SFT (exceeds SFT quality) |
| #362 | M2P at d=1024: 99.6% of SFT (512:1 compression, no cliff) |
| #363 | Layer depth: 99.7% (L=2), 93.5% (L=4), 97.1% (L=8), 86.4% (L=16) |
| #354 | TF-IDF routing: 95% accuracy, 92.2% composition quality |
| #353 | Cross-domain transfer: 8/10 pairs useful, Option A wins |

**Caveat:** All results on 2 valid synthetic domains (sort, reverse) at L=2 toy models.
Natural language, deep models (L=36), and real benchmarks are untested (Levels 1-3 in PoC roadmap).

### Scale & Serving (Proven)

| Finding | Result |
|---------|--------|
| #176 | M5 Pro achieves 73% BW utilization (165.6 tok/s base) |
| #320/#330 | scale=5 gives 0pp MMLU degradation; scale=20 gives -60pp |
| #332 | Full integrated pipeline works on Qwen3-4B-4bit |
| #333 | Expert promotion at scale=5 preserves quality (0pp MMLU) |

### Routing (Supported)

| Finding | Result |
|---------|--------|
| #310 | Per-token hidden states linearly separable at 98.3% |
| #313 | Single-pass MLP routing within 0.61% of oracle |
| #28 | Softmax router matches oracle at N=24 (0% gap, 0% fallback) |

### Permanently Closed Paths

| Path | Why It's Dead |
|------|---------------|
| Weight-space merge into ternary | δ=±1.0 vs needed δ=0.002 (500x too coarse) — #289, #291, #303 |
| SVD solidification for composition | Loses Grassmannian structure: -26pp vs scale reduction — #329 |
| Self-growing from random init | Catastrophic interference at promotion 3 — #331 |
| Room Model (pre-sum W_combined) | Inter-layer nonlinearities kill it — #303, #334 |
| Per-layer routing | Actively harmful, collapses to per-sequence oracle — #29 |
| Binary routing heads | Collapse at N>10, 46% base-only fallback — #27 |
| Sparse-BitNet | Sparse matmul 7% slower on Apple Silicon — #36 |
| KD from large teacher | -34.4% worse than self-supervised — #30 |
| Energy/Gini/spectral routing | NRE is the ceiling, all others killed — spectral arc closed |
| Text-to-LoRA hypernetwork | Tautological kill at N=24 — #33 |
| EigenLoRAx subspace extraction | Grassmannian prevents shared subspace by design — #84 |
| Ternary base advantage | Advantage from ternary ADAPTERS, not base — #52 |

## Active Research: M2P Distillation Pipeline

### The Problem (Finding #341, #342)

M2P with Grassmannian A produces perfect orthogonality (K848 PASS), but B-matrices
collapse to centroid when trained on multiple domains simultaneously. Domains with
low base loss (already-competent) receive adapters calibrated for hard tasks.

- Additive domain conditioning: improved median to 47.3% but didn't break centroid (#342)
- Root cause: M2P attention bottleneck makes embedding gradients low-rank

### What Needs to Happen Next

| Experiment | What It Tests | Status |
|-----------|---------------|--------|
| **exp_m2p_scale_calibrated** | Preservation loss teaches M2P correct scale | Active (P0) |
| **exp_m2p_composition_n5** | 5 M2P-generated adapters compose with Grassmannian guarantee | Open (P0) |
| **exp_m2p_teacher_distillation** | Teacher→student knowledge transfer via M2P (Qwen3-8B → 4B) | Open (P1) |
| exp_shine_architecture_study | Port full SHINE architecture to MLX | Active |
| exp_multi_tenant_serving | Different adapter stacks per user | Active |

### The Fix Strategy

The centroid collapse (#341, #342) is the current bottleneck. Potential fixes:

1. **Multiplicative gating** (not additive) — force M2P attention to domain signal
2. **Per-domain loss normalization** — equalize gradient magnitudes across domains
3. **Separate M2P heads per domain** — break the shared bottleneck
4. **Train on single domain, eval on composition** — sidestep multi-domain training entirely

Each fix must maintain the Grassmannian A guarantee (which is unaffected by training dynamics).

## Capacity Planning

| Scale | Hidden dim | Max orthogonal adapters (d/r at r=16) |
|-------|-----------|--------------------------------------|
| Toy GPT | 64 | 4 |
| BitNet-2B | 2560 | 160 |
| Qwen3-4B | 3584 | 224 |
| Qwen3-8B | 8192 | 512 |

Memory budget on M5 Pro 48GB (Finding #332):
- Base: ~1.2 GB (Qwen3-4B-4bit)
- Per adapter: ~45 MB
- N_max = 853 adapters fit in memory
- At N=500: 23.86 GB (59.6% of budget)

## Two Products

### Pierre Tiny (edge, free tier)
- Base: BitNet-2B or small Qwen
- 5-25 SFT domain adapters
- 73-97 tok/s on M5 Pro
- 1.7-3GB memory
- Per-token softmax routing

### Pierre Pro (cloud, pro tier)
- Base: Qwen3-4B (or promoted)
- 25-200+ domain adapters
- Session adapters (M2P-generated, ephemeral)
- User profile adapters (M2P-distilled)
- Full promotion lifecycle at scale=5
- Per-token routing + block-diagonal attention

## Key References

### Core Architecture
- SHINE (arXiv:2602.06358) — M2P transformer generates adapters from context
- PHATGOOSE (post-hoc gating) — independently trained experts with zero-shot routing
- Grassmannian packing — QR-based orthogonal A-matrices (our approach)
- ReLoRA (arXiv:2307.05695) — repeated merge = full pre-training (promotion foundation)
- FlexMoRE — SVD extraction preserves/improves quality (but kills Grassmannian — #329)

### Composition & Routing
- Naive LoRA Summation (arXiv:2508.11985) — orthogonality enables additive composition
- LoRA Soups (arXiv:2410.13025) — CAT composition beats data mixing
- MoLoRA (arXiv:2603.15965) — per-token LoRA routing
- CLONE (arXiv:2506.02847) — MoE router for dynamic LoRA on edge

### Foundations
- GaLore (arXiv:2403.03507) — low-rank gradient training from scratch
- FlyLoRA (arXiv:2510.08396) — frozen random A as implicit router, JL-lemma
- LeJEPA (arXiv:2511.08544) — SIGReg reasoning chain: make failure impossible with existing math
- BitNet b1.58 (arXiv:2402.17764) — ternary architecture fundamentals

### MLX / Apple Silicon
- MLX-BitNet: exo-explore/mlx-bitnet — first ternary impl for Apple Silicon
- Architecture gallery: sebastianraschka.com/llm-architecture-gallery/
