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
             Compare per-token perplexity     │
             (free signal, no labels needed)  │
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
| Held-out perplexity regression | Indirect | Free | All |

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

## Roadmap

### Phase 1: Prove it works (budget: $50, timeline: 1 week)

1. Build distillation pipeline (Groq batch → QLoRA training)
2. Create 50 domain experts on Qwen2.5-7B
3. Compose all 50 with hash ring routing
4. Benchmark vs base model on standard evals (MMLU subsets, HumanEval)
5. Kill criterion: composed model must beat base on >80% of expert domains

### Phase 2: Prove evolution works (budget: $50, timeline: 1 week)

6. Inject deliberate errors into 5 experts
7. Implement clone-and-compete with shadow scoring
8. Show tournament converges: corrected clone wins in <10K queries
9. Show no regression: original domains unaffected by evolution
10. Kill criterion: tournament must resolve correctly >90% of the time

### Phase 3: Scale (budget: $50, timeline: 2 weeks)

11. Scale to 500 experts across diverse domains
12. Measure hash ring displacement at N=500
13. Run continuous evolution with automated 70B teacher feedback
14. Measure model quality improvement over time without any retraining
15. Kill criterion: quality must monotonically improve over 10 evolution cycles

### Research Questions (open)

- **Optimal clone training budget:** 50 steps? 100? 300? More steps = better
  clone but slower evolution cycle.
- **Tournament sample efficiency:** How many queries to confidently pick a
  winner? Bayesian stopping rule?
- **Cross-expert errors:** When the bug is in composition (not individual
  expert), how does the system detect and fix it?
- **Expert merging:** After v5 beats v4 beats v3 beats v2 beats v1, can we
  merge the accumulated corrections into a single clean expert?
- **Decentralized evolution:** Contributors run their own experts, compete on
  a shared ring, earn reputation/tokens for quality.
