# Research Tree: All Paths, All Evidence, All Gaps

314 findings. 388 experiments. 12 fundamental impossibilities. 3 viable tracks.

## The Proven Foundation (what every track depends on)

These are non-negotiable — proven across hundreds of experiments:

```
Grassmannian orthogonality: cos=0.0002 at d=2560 (#3, #126)
  └── Composition is stable: 0/25 failures, 0 features lost at N=50 (#8, #232)
       └── 1/N scaling prevents catastrophe (#14)
       └── Per-domain scale is mandatory: FORMAT s≤4 vs CAPABILITY s≥20 (#248-250)
       └── NTP >> SFT for reasoning: 30pp gap on GSM8K (#262)
       └── DARE p=0.5 for OOD robustness (#266-269)
```

## 12 Permanently Closed Paths

| # | What's dead | Why (math) |
|---|------------|------------|
| 1 | Merge adapters into ternary weights | 3 levels can't absorb ±1 deltas (#289, #291) |
| 2 | Re-quantize after adding continuous LoRA | Delta 20x grid spacing → destroyed (#289) |
| 3 | Pre-sum W_combined across layers (Room Model) | Inter-layer nonlinearities (#303) |
| 4 | Binary energy gating for LoRA | Capacity always helps → NLL always decreases (#184) |
| 5 | SVD knowledge protection on ternary | Flat spectrum → no low-rank knowledge subspace (#272) |
| 6 | Contrastive decoding with orthogonal adapters | Decorrelated outputs = noise, not signal (#244) |
| 7 | KV-cache reuse across adapter switches | Orthogonal subspaces = incompatible KV (#309) |
| 8 | Post-composition spectral surgery | Surgery suppresses domain-pure signals (#278) |
| 9 | Weight orthogonality → behavioral specialization | Nonlinear mapping; needs data, not structure (#246) |
| 10 | PPL as quality proxy | r=0.08 with task accuracy (#110) |
| 11 | Spectral Gini as composition metric | Production-irrelevant; 6 experiments wasted (#285-286) |
| 12 | Speed beyond factored LoRA (on M5 Pro) | Memory bandwidth ceiling, not compute (#300, #306) |

## The Three Viable Tracks

### Track 1: Production Serving System
**Risk: LOW | Readiness: HIGH | Unique value: PAPER-READY**

Everything proven independently, needs integration:

```
Block-diagonal attention (#314)     ← best single-pass strategy
  + MLP per-token routing (#312-313) ← within 0.61% of oracle
  + Hidden-state probe (#310)        ← 98.3% token accuracy
  + Segment isolation (#305)         ← +16% over per-sequence
  + Per-segment RoPE reset           ← closes 8.9% gap (NOT YET TESTED)
  + Mixed NTP/SFT per domain (#262)  ← NTP for reasoning, SFT for generation
  + DARE p=0.5 (#266)               ← OOD robustness
  + OPLoRA (#271)                    ← best GSM8K + MMLU composition recipe
```

**What's missing:**
- Per-segment RoPE reset (standard technique, not yet implemented)
- Integration test: all components together in one pipeline
- Behavioral eval at N=24 with the combined recipe

**What this produces:** A paper: "Composable Domain Experts with Provable Non-Interference on Ternary LLMs: Single-Pass Serving via Block-Diagonal Attention"

### Track 2: Room Model / Game Engine (Toy Scale)
**Risk: HIGH | Readiness: LOW | Unique value: NOVEL RESEARCH**

Core pre-sum was killed (#303). But the insight has surviving elements:

```
KILLED: W_combined = Σ ΔW_i across all layers (nonlinearities)
ALIVE:  Per-module linearity (MSE 5.6e-7) — within one layer, math works
ALIVE:  Geometric routing concept (token direction = domain selection)
ALIVE:  Adapter gradient as free routing signal (GEOMETRIC_THEORY.md)
ALIVE:  Game engine algorithms map to real problems
```

**What to explore (toy scale on micro/models/gpt/):**
- INTRA-LAYER W_combined (where linearity holds)
- Terrain splatting for per-weight blend maps
- Adapter gradient analysis (∇H as domain structure)
- SHINE-style dynamic context modulation
- Whether nonlinearity can be COMPENSATED (learned correction term)

**What this could produce:** Novel contribution at the intersection of game engine math and neural network composition. But may lead nowhere — that's the risk.

### Track 3: Non-Ternary Base (Qwen3 / Phi)
**Risk: MEDIUM | Readiness: MEDIUM | Unique value: ANSWERS THE BIG QUESTION**

The fundamental open question: is MMLU degradation TERNARY-specific or COMPOSITION-inherent?

```
Evidence ternary is the problem:
  - Flat spectrum violates top-k knowledge assumption (#272)
  - MMLU -5 to -6pp persists through ALL mitigation (#263, #268)
  - OPLoRA removes 99.9% direction interference, recovers only 5pp of 25pp (#272)

Evidence composition is the problem:
  - Composition itself degrades MMLU regardless of training (#263)
  - Wrong adapter still captures 87% of benefit (#203) — adapters are blunt
```

**What to test:**
- Port Pierre (attach_adapter, fit_router, compose_adapters) to Qwen3-4B or Phi-4
- Same Grassmannian init, same NRE merge, same ridge router
- Measure MMLU with composed adapters on fp16 base
- If MMLU degradation disappears → ternary was the bottleneck
- If MMLU degradation persists → composition itself is the bottleneck

**What this produces:** Definitive answer to whether our math works on modern bases. If yes, Pierre becomes architecture-agnostic. If no, we need different math.

---

## The Two Unsolved Fundamental Problems

### Problem 1: Knowledge recall degrades under composition
**Evidence:** #263 (MMLU -5 to -6pp), #268 (persists across ALL DARE rates), #272 (ternary flat spectrum)

**What we've tried that didn't work:**
- DARE sparsification: partial mitigation, doesn't solve (#266-269)
- OPLoRA direction protection: removes 99.9% interference, recovers 5pp of 25pp (#272)
- Spectral surgery: counterproductive (#278)
- Norm equalization: wrong proxy (#285-286)
- Fisher weighting: degenerates to Frobenius (#281)

**What might work:**
- Non-ternary base (Track 3) — eliminates flat spectrum
- Higher rank adapters at lower scale — less perturbation per adapter
- NTP training for ALL domains (not just reasoning) — preserves knowledge (#262)
- Accept it — adapters ARE distribution overrides, not knowledge additions

### Problem 2: Routing collapses at N=24
**Evidence:** 7 methods killed at ~40% (#188-194, #256-257)

**Key finding that changes the framing:** #200 shows routing accuracy is irrelevant for PPL. Every adapter helps every domain. Routing matters only for BEHAVIORAL quality.

**What might work:**
- Per-token routing (#310: 98.3% token accuracy) instead of per-sequence
- Block-diagonal attention for mixed-domain segments (#314)
- Coarser domain groups (reduce N=24 to N=7 genuine clusters)
- Accept it — if every adapter helps, route to any adapter in the right cluster

---

## Gaps We Haven't Explored

| Gap | Why it matters | Blocked by |
|-----|---------------|------------|
| Per-segment RoPE reset | Closes 8.9% gap in block-diagonal (#314) | Nothing — standard technique |
| N=24 with block-diagonal + per-token | Best serving + hardest routing test | Needs #314 + #310 integration |
| Non-ternary composition (Track 3) | Answers the fundamental question | Engineering effort to port |
| SHINE-style dynamic context modulation | Context-aware adapter weighting | Needs SHINE understanding |
| Adapter gradient analysis | Free routing signal from weight structure | Needs experiment |
| Higher-rank adapters (r=32, r=64) | More capacity per domain, less N capacity | Orthogonality check at higher rank |
| Domain hierarchy (medical → cardiology) | Reduces effective N, improves routing | Needs hierarchical adapter training |
| (IA)³ activation scaling instead of LoRA | May avoid the composition-degrades-knowledge problem | Fundamental architecture change |

---

## Recommended Parallel Allocation

| Track | What to do next | Time est. | Who |
|-------|----------------|-----------|-----|
| **1: Production** | Integrate block-diagonal + RoPE reset + per-token routing + DARE + OPLoRA | 4-6 hrs | Ralph loop |
| **2: Room Model** | Toy-scale intra-layer composition + gradient analysis on micro/models/gpt/ | 2-3 hrs | Manual exploration |
| **3: Non-ternary** | Port Pierre to Qwen3-4B, test MMLU with composed adapters | 3-4 hrs | Ralph loop |
| **SHINE** | Clone repo, understand M2P architecture, assess portability | 2 hrs | Manual reading |
