# Research State: 314 Findings, 388 Experiments

## What We Know (the proven foundation)

### Mathematics — PROVEN, non-negotiable
- **Grassmannian orthogonality works**: cos=0.0002 at d=2560, 23x margin at N=24 (#3, #298)
- **Composition is stable**: 0/25 catastrophic failures, self-stabilizing with N (#8, #233)
- **1/N scaling resolves composition catastrophe** (#14)
- **0 features lost even at N=50** topologically (#232)
- **d_crit ~ 32r** for orthogonality to hold (#129) → at r=16, need d≥512 ✓

### Ternary — PROVEN, with known limitations
- **Ternary encodes STRUCTURE not MAGNITUDE** (BITNET_FOUNDATIONS.md)
- **Cannot merge adapters into ternary weights** — 3 levels, no room (#289, #291, #303)
- **Ternary LoRA gives 3-8% BETTER PPL than bf16** (regularization effect) (#290)
- **Flat ternary spectrum violates top-k knowledge assumption** → MMLU math -25pp (#272)
- **Composition itself degrades MMLU -5 to -6pp** regardless of method (#263)

### Routing — PROVEN at N=5, BROKEN at N=24
- **Ridge regression: 96-99.6% at N=5** (#276, #287)
- **Hidden-state probe: 98.3% per-token accuracy** (#310)
- **ALL methods collapse at N=24 (~40%)** — 7 methods killed (#188-194, #256-257)
- **But routing accuracy is IRRELEVANT for PPL** (#200, #203) — every adapter helps all domains
- Routing only matters for **behavioral quality**, not perplexity

### Training — PROVEN, with critical NTP vs SFT finding
- **NTP preserves reasoning** (GSM8K +10pp), SFT destroys it (-20pp): 30pp gap (#262)
- **SFT fixes generative quality** but NOT benchmarks (#180)
- **Per-domain scale is mandatory**: FORMAT (s≤4) vs CAPABILITY (s≥20) regimes (#248-250)
- **DARE p=0.5 partially fixes OOD degradation** (2/5 → 1/5 domains hurt) (#266-269)
- **All 24 SFT adapters converge on BitNet-2B** (17.3% mean improvement) (#297)

### Speed — PROVEN ceiling at ~77 tok/s for side-path
- **Native BitLinear: 140 tok/s** (#288)
- **v3 (bf16 side-path): 73 tok/s** — practical ceiling for side-path (#288)
- **v5 (ternary side-path): 77 tok/s** — best speed with adapters (#290)
- **v6 (precomputed attn-only): 87 tok/s** — fastest but drops code (#292)
- **ALL speed optimizations beyond v5 failed**: precomputed full (bandwidth-bound #299), batched (#306), hybrid (#301), Room Model (#303)
- **The bottleneck is memory bandwidth, not dispatch count** (#300)

### Mixed-Domain Serving — ACTIVE FRONTIER
- **Segment isolation: +16% over per-sequence routing** (#305)
- **Single-pass MLP routing matches multi-pass oracle within 0.61%** (#313)
- **Block-diagonal attention is best single-pass strategy** (#314)
- **KV-cache reuse across adapter switches is HARMFUL** (#309)
- **RoPE position offset is the sole remaining gap** (#314)

---

## What's Permanently Closed (impossibility structures)

| Path | Finding | Why it's dead |
|------|---------|---------------|
| Merge into ternary | #289, #291 | 3 levels, fine-tuning needs δ≈0.002 but ternary forces δ=±1 |
| Pre-sum W_combined (Room Model) | #303 | LayerNorm/softmax/SiLU nonlinearities between layers |
| Spectral Gini optimization | #285-286 | Wrong proxy metric — 6 experiments wasted |
| Verification from hidden states | #293-294 | Correctness signal only in logit space (lm_head) |
| Faster-than-v5 side-path | #299-301, #306 | Memory bandwidth is hardware-fixed |
| Weight orthogonality → behavioral specialization | #246 | Structural orthogonality ≠ behavioral orthogonality |
| KV-cache reuse across adapters | #309 | Cross-adapter KV entries actively harmful |
| Bridge extraction for pathway preservation | #228, #231 | Unnecessary and harmful at our scale |
| PiSSA init | #37 | Incompatible with composable ternary experts |
| Equal-weight composition | #23 | PPL in trillions without 1/N scaling |

---

## The Three Viable Research Tracks

### Track 1: Production Serving (PROVEN, needs engineering)
**Status:** Working system, needs optimization.

**What works:** v3/v5 at 73-77 tok/s, 99.6% routing at N=5, 0.41 behavioral with SFT.

**The frontier:** Block-diagonal attention + MLP per-token routing (#314).
Single-pass serving within 0.61% of multi-pass oracle (#313).

**What's needed:**
- Per-segment RoPE reset (closes 8.9% gap from #314)
- Mixed NTP/SFT adapter selection per domain (#262: NTP for reasoning, SFT for generation)
- DARE p=0.5 for OOD robustness (#266)

**Risk:** Low. All components proven independently. Integration is engineering.

**Unique contribution:** Only system with mathematical non-interference guarantee + per-token routing on ternary base.

### Track 2: Room Model / Game Engine Math (EXPERIMENTAL, needs research)
**Status:** Core idea KILLED (#303) but the insight is valuable.

**What was killed:** Pre-summing W_combined across layers. Nonlinearities between layers destroy linearity.

**What survives:** 
- Per-module linearity IS confirmed (MSE 5.6e-7) — the math works within one layer
- The geometric routing concept (token direction = domain selection) is sound
- Game engine algorithms (terrain splatting, deferred rendering) map to real problems

**What's needed:**
- Reformulate for INTRA-LAYER composition (where linearity holds), not INTER-LAYER
- Toy model experiments on micro/models/gpt/ to test geometric concepts
- Game engine algorithm port: terrain splatting for per-weight blend maps
- SHINE-style dynamic context modulation of blend weights

**Risk:** High. The core assumption was disproven. But the direction has novel elements nobody else is exploring.

**Unique contribution:** If it works, it's the bridge between game engine rendering and neural network composition.

### Track 3: Qwen3 / Modern Architecture (NOT STARTED, needs foundation)
**Status:** Not started. All work so far is on BitNet-2B-4T (2B params, ternary).

**Why consider it:**
- BitNet-2B is the ceiling for behavioral quality (legal 0.066, finance 0.079)
- Ternary flat spectrum fundamentally limits MMLU (#272)
- The field has moved to 7B-1T scale with hybrid attention
- Our composition math (Grassmannian, NRE, ridge router) is architecture-agnostic

**What's needed:**
- Port Pierre (attach_adapter, fit_router, compose_adapters) to a Qwen3 base
- Test: does Grassmannian orthogonality hold on a non-ternary model?
- Test: does the MMLU degradation persist on fp16/4-bit bases?
- If MMLU degradation disappears → the limitation was ternary, not composition

**Risk:** Medium. The math should transfer. The engineering effort is moderate.

**Unique contribution:** If composition works on Qwen3 without MMLU degradation, the ternary limitation is identified and the architecture becomes competitive.

---

## The Two Unsolved Fundamental Problems

### Problem 1: Composition degrades knowledge recall (MMLU -5 to -6pp)
**Evidence:** #263, #268, #272
**Root cause:** Ternary flat spectrum means knowledge is encoded diffusely, not in top-k singular vectors. Composition averages this diffuse signal.
**Status:** No known fix. DARE partially mitigates. OPLoRA removes 99.9% direction interference but only recovers 5pp of 25pp gap.
**Question:** Is this ternary-specific or composition-inherent? Track 3 (Qwen3) would answer this.

### Problem 2: Routing collapses at N=24
**Evidence:** #188-194, #256-257, #298
**Root cause:** Semantic domain overlap (economics≈finance, psychology≈sociology). Not an architecture problem.
**Status:** Ridge router gets 42.1%, but hidden-state probe gets 98.3% per-token (#310). The solution is per-token routing, not per-sequence.
**Question:** Does block-diagonal + per-token routing (#314) solve this at N=24? Not yet tested.

---

## Recommended Parallel Work Allocation

| Track | Effort | Risk | Unique value | Priority |
|-------|--------|------|-------------|----------|
| **1: Production serving** | Low (engineering) | Low | Working system, paper-ready | P0 |
| **2: Room Model / game engine** | Medium (research) | High | Novel direction, unexplored | P1 (toy scale) |
| **3: Qwen3 composition** | Medium (porting) | Medium | Answers the ternary question | P1 |

All three can run in parallel. Track 1 produces the paper. Track 2 explores the novel direction. Track 3 resolves the fundamental question about ternary vs composition.
