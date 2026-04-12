# Vision: Pierre — Composable Domain Intelligence

## Core Thesis

A language model is not a monolith — it is a frozen base plus composable domain experts.
Each expert is a lightweight adapter that adds domain knowledge, trained in 15 minutes,
composable without interference, and shareable through a flywheel that makes the base
model smarter from usage.

**Platform:** Gemma 4 E4B 4-bit (open, free).
**Serving:** Together AI serverless ($0.10/M tokens) or local (M-series Mac, $0).
**Adapters:** PoLAR r=6 on v_proj+o_proj. 5MB each. Zero-interference composition.

## What Pierre Is

A coding agent (like Claude Code) where every user trains the model:

```
User works on project → conversation trains personal adapter (1.2 min)
                       → adapter persists across sessions (1.25MB file)
                       → user shares adapter → flywheel improves base for everyone
```

The adapters don't just "nudge" — they add real domain knowledge (math: +72pp,
medical: +22pp, legal: +90pp format compliance) and compose without interference.

## Architecture (Proven by 600 Experiments)

```
┌─ Base: Gemma 4 E4B 4-bit (frozen, shared) ───────────────────┐
│                                                                │
│  ┌─ Domain Adapters (PoLAR r=6, v_proj+o_proj) ────────────┐ │
│  │  Pre-merged into base for top-5 domains (0ms overhead)   │ │
│  │  Grassmannian A-matrices: zero interference (cos=1e-16)  │ │
│  │  PoLAR: full rank utilization (sr=6.0 exact)             │ │
│  │  Null-space restriction: 98.7% quality, base untouched   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─ Dynamic Adapter (per request, routed) ──────────────────┐ │
│  │  TF-IDF ridge router: 96% at N=5, 84% at N=25           │ │
│  │  Swap: 1ms hot, 5ms cold. Throughput: 96% of base.       │ │
│  │  200+ niche domains in adapter CDN                        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─ Personal Adapter (per user, persistent) ────────────────┐ │
│  │  Rank-16, trained from 50 examples in 1.2 min            │ │
│  │  Online learning: +60pp from 20 conversation turns        │ │
│  │  Domain-conditional: retrained on domain-fused base       │ │
│  │  Survives context window clearing                         │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌─ Format Constraints (per domain, at decode time) ────────┐ │
│  │  XGrammar: Python CFG, SOAP, legal citation, JSON schema │ │
│  │  Think-then-constrain: reason freely, then enforce format │ │
│  │  Zero syntax errors on constrained output                 │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

## The Flywheel

```
User trains adapter → shares to registry → cluster similar adapters
      ↑                                           ↓
      │                                    crystallize into domain adapter
      │                                           ↓
      └── improved base ← promote into base (ε=4.78%, 3 cycles proven)
```

Every user interaction makes the system smarter. Competitors can't copy the
flywheel — they'd need the user base.

## What We Proved (600 experiments, 504 findings)

### Composition (zero interference)
- Grassmannian orthogonality: cos=1.7e-16 (Finding #341, conclusive)
- N=100 composition: max_cos=2.25e-8 (Finding #440)
- Pre-merge: lossless for orthogonal adapters (0ms overhead)
- Cross-projection (#483): weight-space composition kills quality → use DCCD instead

### Adapter Quality
- PoLAR r=6: 72% GSM8K, sr=6.0 exact (Finding #442, C1.1v2)
- v_proj+o_proj: +70pp SOAP, +90pp legal format (Finding #480)
- Online learning: +60pp from 20 conversation turns (Finding #490)
- 4-bit quantization: lossless 3x compression (Finding #422)

### Serving
- Swap: 1ms hot, zero graph recompilation (Finding #503)
- Throughput: 96% of base (Finding #435)
- Format compat: MLX ↔ PEFT bijection verified (Finding #481)
- Hot-add/remove: O(1), bit-exact (Findings #429, #430)

### Null-Space Isolation
- 512 dims available in local q_proj (85 adapter slots) (Finding #493)
- 2048 dims in v_proj (341 slots) (Finding #493)
- 98.7% quality preserved with exact base orthogonality (Finding #494)

### Permanently Closed Paths (28 killed)
- M2P/SHINE for document QA (centroid trap, Finding #345/#486)
- Direction-only adapters (magnitude matters through FFN, Finding #439)
- LoRI sparse masks (wrong orthogonality level, Finding #487)
- Spectral surgery on PoLAR (flat spectrum, Finding #488)
- TTT zero-cost update (impossible for transformer LoRA, Finding #491)
- Self-organizing null-space slots (structurally impossible, Finding #501)
- Simultaneous adapter activation without routing (catastrophic, Finding #425)

## The Competition

| | Claude Code | Pierre Phase 1 | Pierre Phase 3 |
|---|---|---|---|
| Base quality | Excellent | Good (Gemma 4) | Good + carved experts |
| Domain expertise | Generic | 5 domains | 100+ domains |
| Personalization | None | Per-project | Per-user, self-learning |
| Cost/request | ~$0.01 | ~$0.00005 | ~$0.00002 |
| Improves from usage | No | No | Yes (flywheel) |
| Works offline | No | Yes (Tier 3+) | Yes + federated |

## Research Frontier (P9, 20 experiments queued)

- CMoE: carve Gemma 4 into sparse experts (2.4x faster, free extraction)
- TT-LoRA: 180KB adapters (28x smaller than current)
- DES: test-time search over expert count (benchmark improvement)
- MemoryLLM: self-updating weight memory (no gradients)
- Sigmoid routing: multi-expert activation (MiniMax pattern)
- Self-evolution: model improves its own scaffold
- CISPO RL: preserve rare reasoning token gradients

## Key References

- PoLAR (arXiv:2506.03133) — polar-decomposed adapter, Stiefel manifold
- Null-LoRA (arXiv:2512.15233) — null-space restriction for base isolation
- CMoE (arXiv:2502.04416) — carve dense model into MoE in 5 min
- TT-LoRA (arXiv:2504.21190) — tensor train adapters, 33K params
- ReMix (arXiv:2603.10160) — RL routing prevents softmax collapse
- SHINE (arXiv:2602.06358) — context-to-parameter hypernetwork (killed for QA, valid for compression)
- MiniMax M1 (arXiv:2506.13585) — 256-expert MoE, CISPO RL, self-evolution
