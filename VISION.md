# Vision: Composable Ternary Experts

## Core Thesis

A language model is not a monolith — it is a scaffold plus composable experts.
Each expert is cheap ($0, local), independently trainable, hot-swappable, and
structurally guaranteed not to interfere. The more contributors, the better
the model. No retraining. No GPU. Commodity hardware.

## Architecture: BitNet-SOLE

```
Ternary Base (BitNet-2B-4T, or GaLore-grown scaffold)
    │
    ├── Instruction Adapter (always composed, chat/QA behavior)
    ├── Per-token Router (selects top-k domain experts)
    │     ├── Domain Expert 1 (ternary LoRA, rank-16, ~1.9KB)
    │     ├── Domain Expert 2
    │     └── ... N experts (proven to N=25)
    └── Capability Adapters (reasoning, safety — on demand)

Skeleton: Grassmannian AP-packed frozen A matrices
         → 17x decorrelation filter on B-matrix interference
         → plug-and-play guarantee: add/remove expert = pointer change

Serving: bf16 merge for always-on adapters (16.7 tok/s, 1.7GB)
         Runtime LoRA for on-demand experts (12.3 tok/s at N=5)
         llama.cpp --lora for multi-adapter CPU serving (33.8 t/s)
```

### Why Ternary

| Property | FP16 | Ternary |
|----------|------|---------|
| Composition ratio (N=5) | PPL trillions (unscaled) | 3.45x (stable) |
| Adapter cosine at convergence | 0.142 (Qwen-7B) | 0.00125 (BitNet-2B) |
| Adapter storage | 18.4 KB | 1.9 KB (10x smaller) |
| Composed PPL (ternary adapters) | 4.35 | 4.16 (-4.4% better) |
| Serving | GPU required | CPU commodity hardware |
| Merge into base | Works | bf16 merge works (7% better PPL than runtime LoRA) |

### The Grassmannian Skeleton (Plug-and-Play Guarantee)

Pre-computed orthonormal A matrices on Gr(r, d) via Alternating Projection.
Frozen during training. The skeleton guarantees:

- ||ΔW_i^T ΔW_j|| ≤ (α/r)² · ||B_i|| · ||A_i^T A_j|| · ||B_j||
- If A_i ⊥ A_j: interference → 0 regardless of B correlation
- Empirically confirmed: B-matrix cos 0.0298 → delta cos 0.0017 (17x filter)
- Capacity: N_max = d²/r² (25,600 at d=2560/r=16)

## What We Proved (23 BitNet experiments)

### Conclusive

| Finding | Result |
|---------|--------|
| Orthogonality holds at convergence | |cos|=0.00125, 40x below 0.05 threshold (PROVEN) |
| Composition reproducible | CV=0.5% across 3 seeds (PROVEN) |
| SOLE routed beats monolithic at matched params | 4/5 domains, -3.7% avg PPL at 108M parity |
| Composition scales to N=25 | gamma=0.982, all 25 domains benefit |
| Reasoning composes without interference | 5/5 domains improved, mean interference -1.49% (beneficial) |
| Instruction tuning fixes task eval | K1 PASS (1/4 worse), math +6.7pp |
| Ternary adapters compose better than FP16 | -4.4% PPL, -19.3% cosine (3 seeds) |

### Killed (important negative results)

| Finding | Implication |
|---------|-------------|
| Weight-space orth != data-space orth (OSRM) | 100% pairs fail OSRM (<0.1), mean ratio 0.86. Yet composition WORKS (4/5 pairs). Constructive transfer, not orthogonality, is the mechanism. |
| Clone-compete evolution: warm-start = cold-start | Evolve ≠ clone-compete. Evolve = retrain-from-scratch + quality gate. |
| LoTA-QAF ternary merge impossible (116x gap) | bf16 float merge works (7% better PPL), runtime LoRA for dynamic routing |
| Base-free scaffold: PPL 319M (pretrained adapters), PPL 186-2887 (fresh adapters) | Pretrained base is essential. Even fresh adapters trained ON scaffold hit capacity limit (36-642x gap). |
| NTP adapters fail task eval (3/5 worse) | Instruction-format training mandatory |
| Ternary base doesn't improve orthogonality | Advantage is from ternary ADAPTERS, not ternary base |
| EigenLoRAx subspace extraction fails (+80.8% PPL gap) | Grassmannian A-matrices prevent shared subspace. Orthogonality enables composition but prevents cross-adapter transfer. Evolve = retrain-from-scratch only. |

## Readiness Assessment

| Phase | Readiness | Status |
|-------|-----------|--------|
| Distill | **70%** | Instruction-format works. KR-Test eval done (rho=1.0 rank signal). |
| Compose | **55%** | N=25 scales. Matched-param wins. Top-2 routing +13.9% over uniform (per-sequence, 659K router). |
| Evolve | **10%** | Clone-compete killed. KR-Test quality gate metric available (delta>0.03). Retrain design needed. |
| Serve | **35%** | bf16 merge (16.7 tok/s) + runtime LoRA (12.3 tok/s). llama.cpp proven (33.8 t/s). |
| Base-free | **10%** | Random scaffold killed. GaLore/meta-scaffold unexplored. |
| **Overall** | **~35%** | Composition + routing validated. Serving + evolve + base-free are gaps. |

## Active Research Tracks

### Track 1: Foundation Fixes (P1)
- Fix orthogonality.py to measure effective delta vec(B@A)
- ~~KR-Test evaluation~~ DONE: delta rank-correlates with task accuracy (rho=1.0), K2 marginal
- LoRI B-sparsification (frozen A + 90% sparse B, COLM 2025)

### Track 2: Base-Free Scaffold (P1)
- Train fresh adapters ON random scaffold (untested — prior test used wrong adapters)
- GaLore ternary base from scratch (arxiv 2403.03507, <1% gap at 1B)
- Meta-learned scaffold via MAML (genuinely novel, no prior work)

### Track 3: Production Serving (P2)
- llama.cpp --lora on BitNet GGUF (multi-adapter CPU serving)
- Per-token routing (MoLoRA style, arxiv 2603.15965)
- Docker packaging + stress test on Apple Silicon

### Track 4: Evolve Redesign (P2)
- Retrain-from-scratch with combined/better data (winner from clone_compete_powered)
- Quality gate: KR-Test + domain benchmark + composition regression
- Contributor reputation via adapter ELO on held-out eval

## Key References

- GaLore (arxiv 2403.03507) — low-rank gradient training from scratch, <1% gap at 1B
- MoLoRA (arxiv 2603.15965) — per-token LoRA routing, 1.7B+4 adapters beats 8B
- LoRI (arxiv 2504.07448) — frozen A + 90% sparse B, 17.3% better merge
- KR-Test (arxiv 2601.03505) — knowledge retention eval via contrastive examples
- Naive LoRA Summation (arxiv 2508.11985) — orthogonality enables additive composition
- Cornerstone Layers (arxiv 2409.14381) — layer criticality explains base-free kill
- IKnow (arxiv 2510.20377) — instruction-format CPT for domain adaptation
- LoRA Soups (arxiv 2410.13025) — CAT composition beats data mixing
- BitNet b1.58 (arxiv 2402.17764) — ternary architecture
- MoTE (arxiv 2506.14435) — frozen shared + ternary routed experts
- OSRM (arxiv 2505.22934) — data-aware orthogonality fixes weight-only orthogonality gap, +12.78%
- FlyLoRA (arxiv 2510.08396) — frozen random A as implicit router, JL-lemma orthogonality
- Spectral Surgery (arxiv 2603.03995) — training-free LoRA refinement via SVD reweighting
- EigenLoRAx (arxiv 2502.04700) — recycle adapters into principal subspace, 100x fewer params
- ZKLoRA (arxiv 2501.13965) — zero-knowledge proof for LoRA verification, enables marketplace
