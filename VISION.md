# Vision: The Living Composable Model

## Core Thesis

A language model doesn't need to be retrained. It grows by acquiring experts,
improves by evolving them, and scales by adding capacity — all at inference time.

Three pillars, each proven:

1. **Orthogonality is free** — LoRA adapters can't interfere (cos=0.0002 at d=896)
2. **Composition is plug-and-play** — hash ring routing, zero recalibration needed
3. **Evolution through competition** — clone, correct, compete, prune

## What We Proved

| Finding | Result | Implication |
|---------|--------|-------------|
| LoRA orthogonality | cos=0.0002, 50x better than theory | Adapters can't interfere |
| MoE beats joint training | -0.70% vs joint (equalized compute) | Composition > retraining |
| Hash routing plug-and-play | 5.3% displacement at N=20 | No recalibration needed |
| L2 norm stability | 0/25 catastrophic failures | Composition is safe |
| Prune-then-compose | +0.012% gap | Contributors work independently |
| Latency overhead | 0.98% theoretical, <5% with fused kernels | Near-free composition |
| FFN-only KILLED at macro | PPL +66.7%, ortho 424% worse independently trained | All-modules adapters required; attention regularizes FFN diversity |
| Attention amplifies domain overlap | cos=0.85 vs 0.59 (math-med) | Attention is composition risk factor but essential for quality |
| Adapter taxonomy: LoRA optimal | FIT=0.875, 15 types, 3 composition classes | LoRA confirmed best for composable architecture |
| Base-freedom theoretically possible | ReLoRA/LTE achieve full-rank from LoRA | Base model could be expressed as composable adapter |
| ReLoRA base supports composition | cos ratio 1.77x (p=0.056 n.s.), loss ratio 1.052 | Base-freedom path empirically viable |
| Base decomposable into adapter | rank-16 SVD: loss ratio 1.014, cos 1.22x (3 seeds) | Entire model is composable: skeleton + base adapter + N experts |
| Expert quality degrades slower than base | rank-8: base -10%, experts -5% | Composition is robust to base perturbation |
| Zero-shot base transfer works | rank-16: 4.2% loss, rank-32: 0.3% loss | Base swapping without expert retraining is viable |
| Inference latency N-independent | pre-merge +2.6% max, dynamic O(k) not O(N) | Expert library size doesn't slow inference |
| Architecture named SOLE | Structurally Orthogonal Latent Experts; 13-paper survey | Clear terminology for papers and communication |
| Domain similarity predicts collisions | within-cluster \|cos\| 7.84x higher than cross (15 experts, 3 clusters) | Collision landscape is predictable, not random |
| Full-sequence PPL ≠ task quality | Pearson r=0.08; reverse expert: PPL -27% but Acc +9.5pp | Shadow scoring needs answer-conditioned PPL |
| Answer-conditioned PPL works | Pearson r=0.811 vs full-seq r=-0.31 (3 seeds) | Shadow scoring metric validated for Evolve |
| Pre-merge vs dynamic vacuous at micro | 0% specialization → 0% gap; methodology validated | Needs macro retest with real expert specialization |
| Content-aware routing killed at micro | Best 26.5% < 60% threshold; cluster-level trivial (96%) | Hash ring sufficient; hierarchical routing possible |
| Collision rate decreases with N | beta=-0.575, only 1.23% at N=20 (24x below kill) | Composition gets safer as more experts added |
| SOLE positioned vs LoRA Soups | 5 structural advantages; complementary not competing | Clear differentiation from closest prior work |
| 50-expert distillation pipeline | 98% win rate, 42.2% avg PPL improvement (contaminated eval), $0.44/expert | Distill phase directionally validated; MMLU/HumanEval pending |

Scaling law **N_max = d^2/r^2** validated:

| Base Model | d | Composable Experts (r=16) |
|------------|---|-------------------------:|
| Qwen 0.5B  | 896 | ~122K |
| Qwen 7B    | 4096 | ~609K |
| Qwen 72B   | 8192 | ~2.4M |

## The Three Phases

### Phase 1: Distill — Create Experts at Scale

Experts are distilled from large teacher models into small LoRA adapters.
A 70B teacher produces training data; a 7B student learns it as a rank-16 LoRA.

```
Teacher (70B, via Groq/Cerebras API)
    │
    │  generates 1,000 domain-specific instruction-response pairs
    │  per domain ($0.19/expert via batch API)
    │
    ▼
Student (7B base, frozen)
    │
    │  QLoRA fine-tuning, 300 steps, ~15 min on A5000
    │  produces rank-16 adapter (~6MB)
    │
    ▼
Expert registered on hash ring
    │
    │  compose add expert.safetensors --name python-async
    │  immediately receives ~1/N of traffic
    │
    ▼
Serving (no retraining, no recalibration)
```

**Economics at scale:**

| Scale | Data Cost | Training Cost | Total | Per Expert |
|-------|-----------|---------------|-------|------------|
| 50 pilot | $10 | $3 | $13 | $0.26 |
| 500 | $95 | $32 | $127 | $0.25 |
| 5,000 | $950 | $320 | $1,270 | $0.25 |
| 100,000 | $19K | $6.4K | $25K | $0.25 |

A 7B model with 5,000 domain experts: stored knowledge equivalent to
7B + 5,000 * 3.1M = 22.5B parameters. Active cost: 7.003B params per token.

### Phase 2: Compose — Serve with Routing

```
Input tokens
    │
    ▼
┌─────────────┐
│  Base Model  │  Frozen Qwen2.5-7B (shared by all)
│  (frozen)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Router                                 │
│  • Hash ring: zero-shot, plug-and-play  │
│  • Softmax: calibrated, best quality    │
│  Select top-k=2 experts per token       │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  Expert Library (NVMe)                  │
│  N adapters, only k=2 active per token  │
│  Each: rank-16 LoRA, ~6MB              │
└──────┬──────────────────────────────────┘
       │ weighted sum of adapter outputs
       ▼
   Output logits
```

**vLLM handles serving.** Our compose CLI manages the expert registry and
hash-ring routing. vLLM's fused MoE-LoRA kernel handles the inference.

### Phase 3: Evolve — Learn Without Retraining

This is what makes the model *living*. When an expert produces a wrong answer,
instead of retraining in-place (which risks forgetting), we **clone and compete**:

```
                    ┌─── correction arrives ──┐
                    │                         │
                    ▼                         │
  Expert v1 ── Clone ──→ Expert v2            │
  (untouched)        (fine-tuned with fix,    │
                      50-100 steps, ~30 sec)  │
                    │                         │
                    ▼                         │
             Both serve on hash ring          │
             Both see similar queries         │
                    │                         │
                    ▼                         │
             Shadow scoring:                  │
             When v1 selected, also score v2  │
             Compare answer-conditioned PPL   │
             (full-seq PPL killed as proxy)   │
                    │                         │
                    ▼                         │
             After ~1K-10K queries:           │
             Clear winner emerges             │
                    │                         │
               ┌────┴────┐                    │
               ▼         ▼                    │
          v2 wins    v1 wins                  │
          prune v1   prune v2                 │
               │         │                    │
               └────┬────┘                    │
                    │                         │
                    ▼                         │
             Next correction ────────────────┘
```

**Why this works:**

- **No forgetting risk.** Original expert is never modified. If the clone is
  worse, discard it. Zero downside.
- **No differentiable routing needed.** Hash ring places both experts; real
  traffic is the test. No gradient computation for the tournament.
- **Orthogonality headroom absorbs clones.** 600K slots means 1,000 active
  experts + 100 competing clones uses 0.18% of capacity.
- **Quality signal is free.** Next-token perplexity on real queries. No labels,
  no human feedback required (though human feedback accelerates it).
- **Expert lineage is traceable.** python_v1 → python_v2 → python_v3. Each
  generation carries the corrections that survived competition.

**Correction sources (automated):**

| Source | Quality | Cost | Domains |
|--------|---------|------|---------|
| Unit test execution | Perfect | Free | Code |
| 70B teacher judges 7B output | Good | $0.001/query | All |
| Self-consistency (sample N) | Medium | N * inference | All |
| Human feedback | Highest | Expensive | Critical domains |
| Answer-conditioned PPL regression | Indirect | Free | All |

**Expert lifecycle controls:**

- Max 3 concurrent clones per domain (prevents bloat)
- Tournament timeout: 10K queries (conservative default)
- Periodic lineage cleanup: retrain from accumulated corrections
- Automatic regression detection: flag experts whose perplexity drifts >5%

## The Full Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVING COMPOSABLE MODEL                   │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ DISTILL  │───→│ COMPOSE  │───→│  EVOLVE  │──┐           │
│  │          │    │          │    │          │  │           │
│  │ Teacher  │    │ Hash ring│    │ Clone &  │  │           │
│  │ → LoRA   │    │ routing  │    │ compete  │  │           │
│  │ experts  │    │ + vLLM   │    │ + prune  │  │           │
│  └──────────┘    └──────────┘    └──────────┘  │           │
│       ▲                                        │           │
│       │              feedback loop             │           │
│       └────────────────────────────────────────┘           │
│                                                             │
│  Properties:                                                │
│  • Never retrain the base model                             │
│  • Add knowledge by adding experts ($0.25 each)             │
│  • Fix mistakes by cloning + competing (30 sec + traffic)   │
│  • Remove knowledge by pruning experts (instant)            │
│  • Scale: 7B base → 600K experts → 1.86T stored params     │
│  • Active cost: 7.003B params per token                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Roadmap — Execution Cycles

See `ORCHESTRATION_PLAN.md` for full cycle-by-cycle plan with gate criteria.

### Current State: ~55% macro readiness

| Pillar | Micro | Macro | Readiness |
|--------|-------|-------|-----------|
| Orthogonality & composition | Strong | Strong (d=896) | **90%** |
| Adapter type | All-modules confirmed (FFN-only killed at macro) | Decision locked | **100%** |
| Inference scaling | Proven N-indep | Partial (k=1 only) | **50%** |
| Routing | Hash ring validated, content-aware killed | Pre-merge sufficient at small N | **40%** |
| Distillation pipeline | N/A (macro-only) | **SUPPORTED** (98% win, 42.2% avg improvement on contaminated eval, $0.44/expert; MMLU/HumanEval pending) | **70%** |
| Evolution / clone-compete | Answer-conditioned PPL proven (r=0.811) | Not tested | **15%** |
| Base-freedom | Micro proven, delta_rank revise | In progress (ReLoRA macro) | **30%** |

### Cycle 1 (COMPLETE — CPU; PARTIAL — GPU)
- CPU DONE: content-aware routing (killed), pre-merge vs dynamic (killed), answer-conditioned PPL (proven), delta rank scaling (revise), SOLE vs LoRA Soups (proven), collision scaling (supported)
- GPU DONE: FFN-only matched rank (killed — PPL +66.7%, ortho 424% worse)
- GPU DONE: distillation pilot 50 (SUPPORTED — 98% win rate, 42.2% avg improvement on contaminated eval; MMLU/HumanEval pending)
- GPU REMAINING: GPU latency validation (running), ReLoRA macro

### Cycle 1.5 (RUNNING)
- CPU: Cycle 2 micro experiments (composition vs monolithic, cross-domain, shadow scoring)
- GPU: Finishing Cycle 1 critical path (distillation pilot 50) + Cycle 2 GPU tasks

### Cycle 2: Close Phase 1 — Distill
- Scale to 500 experts, teacher size comparison, merge pipeline
- Baselines: composition vs monolithic, synthetic vs real data
- **Gate:** 50-expert model beats base on >80% of domains **[PASSED: 98% on contaminated eval; downstream task eval pending]**

### Cycle 3: Close Phase 2 — Compose + Serve
- Content-aware routing at scale, FAISS indexing
- vLLM production serving, expert add/remove, quantization
- **Gate:** serving with <5% overhead, routing >80% accuracy at N=500

### Cycle 4: Close Phase 3 — Evolve
- Shadow scoring, clone-and-compete, evolution convergence
- Correction pipeline, model collapse detection
- **Gate:** tournament resolves >90%, monotonic improvement over 10 cycles

### Cycle 5: Base-Freedom (stretch)
- ReLoRA from scratch, LTE parallel construction, full base-free pipeline
- **Gate:** entire model is composable adapters, no sacred weights

### Open Research Questions

- **Pre-merge dilution at large N:** At N=500, each expert is 0.2% strength. Does quality survive?
- **Optimal clone training budget:** 50 steps? 100? 300?
- **Tournament sample efficiency:** Bayesian stopping rule for clone-compete?
- **Cross-expert errors:** Bugs in composition, not individual expert — how to detect?
- **Expert merging:** After v5 beats v1, can we consolidate accumulated corrections?
- **Decentralized evolution:** Contributors run own experts, compete on shared ring
- **Delta rank scaling:** Ratio decreases (0.664 -> 0.538 over d=64 to d=256, power law d^(-0.15)). Revised with convergence control — K1 killed but r_95 metric promising. See micro/models/delta_rank_scaling/
- **SOLE vs LoRA Soups:** Resolved (proven). 5 structural advantages identified. Complementary, not competing.
- **CAT weight convergence:** Do CAT-optimized weights converge to ~1.0 at macro scale? Falsifiable test of SOLE theory. Needs pilot 50 experts.
- **LoRA-Flow comparison:** Missing from SOLE positioning. Dynamic per-layer weights — superset of SOLE and CAT. Literature review needed.
- **Held-out eval for pilot 50:** Current 42.2% improvement is on contaminated eval. MMLU subsets + HumanEval needed to upgrade from "supported" to "proven".
- **Composition quality at scale:** Do N=50 pre-merged experts maintain individual quality? Key gate for scaling to 500+.
