# Vision: Composable Ternary Experts

## Core Thesis

A language model is not a monolith — it is a scaffold plus composable experts.
Each expert is cheap ($0, local), independently trainable, hot-swappable, and
structurally guaranteed not to interfere. The more contributors, the better
the model. No retraining. No datacenter GPU. Runs on Apple Silicon.

## Target Platform (Hard Constraint)

**Apple M5 Pro, 48GB unified memory.** This is the deployment target, not a
stepping stone. All training, composition, and serving must fit within this
hardware envelope (~40GB usable). If it doesn't run on the best consumer chip,
it doesn't meet the vision. RunPod/CUDA is for validation comparisons only.

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

Serving: Native BitLinear + runtime LoRA (172 tok/s base, 97 tok/s addmm adapter, 1.22GB at N=1)
         Scales to 25 adapters at 2.26GB, 50 at 2.79GB
         bf16 merge for always-on adapters (16.7 tok/s, 1.7GB)
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
| Competitive benchmark: SOLE not competitive with Qwen2.5-3B | Loses 4/6 benchmarks. Beats Gemma-2-2B 6/6. Memory 2.2-4.5x Qwen. Uniform composition hurts math/legal. Routing + larger base needed. |
| Falcon-E-3B: base quality was the gap, not composition | Falcon-E-3B BASE beats Qwen-3B-4bit 5/6 without adapters. Uniform composition degrades 5/6 benchmarks vs base (confirms routing mandatory). K3 FAIL: 8.80GB memory (bf16 unpack). The competitive gap was base model quality, not SOLE architecture. |
| Per-layer pointer routing hurts (per-sequence is correct) | Per-layer adapter mixing is actively harmful: hash -1.1%, MLP -0.5% vs uniform. Winning strategy (learned gate +11.9%) collapses to same-adapter-all-layers (per-sequence oracle). Adapters calibrated for same-adapter residual stream at all preceding layers — mixing breaks coherence. Per-SEQUENCE routing confirmed as correct granularity. |
| Binary routing heads collapse at N>10 | gamma_routed degrades 0.668→0.851 as N=5→24 (46% base-only fallback). Oracle gamma improves 0.668→0.625 — architecture works, routing is the bottleneck. Random routing beats trained routing at N≥10. |
| Softmax router matches oracle at N=24 | Softmax router: gamma 0.625 = oracle (0.0% gap) despite 40% classification accuracy. Eliminates binary head collapse (0% fallback). Within-cluster misrouting is quality-benign. Random routing 11.6% worse — router provides genuine semantic-cluster value. Binary heads are dead. |
| Depth routing cannot improve token-only routing | Per-layer depth routing (pseudo-queries + expert embeddings) stays near-uniform (entropy 0.992) and degrades quality -18.3% vs token-only. Token routing already at oracle; no headroom for second routing axis. Mixed domain blow-up from steep norm gradient (2.8x). L=4 too shallow. Dead end when token routing is optimal. |
| KD from large teacher fails for ternary adapters | Sequence-level KD from Qwen2.5-7B-Instruct: -34.4% worse than self-supervised (0/5 domains). Distribution mismatch (verbose teacher vs terse eval). Lower training loss + higher eval PPL = learning wrong distribution. Cross-tokenizer sequence-level KD dead for this use case. |

## Readiness Assessment

| Phase | Readiness | Status |
|-------|-----------|--------|
| Distill | **70%** | Instruction-format works. KR-Test eval done (rho=1.0 rank signal). **Training profiled:** batch=1 optimal for 200-step regime (21s/adapter). Batch=8+compile: 7.52x throughput for large datasets. Python tricks negligible (<2%). |
| Compose | **87%** | N=25 scales. N=24 real-data adapters: all-24 uniform -29.1% vs base (constant memory 17.1GB, linear time). Matched-param wins. **ROUTING SOLVED:** Softmax router matches oracle quality (gamma 0.625 = oracle, 0.0% gap) at N=24 despite 40% classification accuracy — within-cluster misrouting is benign. Binary sigmoid heads dead (46% fallback). Softmax: 0% fallback, 330K params (6x less). Random routing 11.6% worse — router provides genuine semantic-cluster value. Hot-add validated: +0.70% avg degradation at N=5→6, zero retraining. Real HF data: correct multi-A composition -26.3% avg PPL at N=5, -29.1% at N=24. Per-token Gumbel-sigmoid routing works on MLX (0.58% overhead, diversity 2.42). Orthogonality stable at N=24: mean \|cos\|=0.0238 (6.7x below N_max=160). **Composition landscape convex:** smooth monotonic curves, mixing beats selection 3.5-5.2%, uniform 1/N only 0.7% from optimal. Gradient-based weight optimization viable. Caveat: PPL-benign, task-specific metrics untested. |
| Evolve | **20%** | Clone-compete killed. ELO tournament killed (ranks composition compatibility, not individual quality). KR-Test quality gate metric available (delta>0.03). **Pruning validated:** LOO removes 5/24 (20.8%) with +0.43% same-domain impact. Four pruning metrics capture different signals (Jaccard ≤ 0.25). Oracle PPL spread 6.2x — quality driven by base model per-domain capability, not adapter training. Retrain design still needed. |
| Serve | **80%** | **Memory-optimized serving: 1.22 GB total (native BitLinear + runtime LoRA), 172 tok/s base, 97 tok/s with adapter (addmm), scales to 25 adapters at 2.26 GB.** Prior 82 tok/s was measurement artifact (Python overhead); prior 10.98 GB was bf16 unpack bug — both fixed. addmm fusion: +10% free speedup. Attn-only LoRA: 127 tok/s (quality unvalidated). Pre-merge WORSE for ternary (-36%, destroys BW advantage). bf16 merge (16.7 tok/s) + llama.cpp (33.8 t/s). Ternary B-matrix: 15.8x adapter compression, pure-addition composition enabled. **E2E pipeline validated:** full query→entropy→route→compose→generate works at 38ms/tok interactive, +44% mean PPL improvement (all 5 domains). K1 FAIL: 2.33x latency from ternary→dense weight conversion (open problem, not architectural). **Batched pre-merge throughput:** runtime LoRA dominates pre-merge by 4-87x for per-token routing (factored O(d*r) vs materialized O(d*r*d)). Architectural split confirmed: pre-merge for always-on adapters, runtime LoRA for routed experts. **Composition latency sweep:** pre-merge scales linearly to N=100 (3.28ms compiled at N=25, 15x under 50ms budget). mx.compile 2.4x speedup for pre-merge only. **mx.compile redundant for generation** — async_eval already hides dispatch; bottleneck is memory bandwidth (74.2% of 273 GB/s), not Python dispatch. **Memory budget validated:** N_max=853 adapters on 48GB M5 Pro (base 1.18 GB = 3% of budget, per-adapter 45.2 MB, routing head 82 KB). At N=500: 23.86 GB (59.6%). **Pair-level weight caching DEAD:** C(50,2)=1225 pairs too many for LRU (3.9% hit rate on balanced traffic), 2.5 GB/pair at production scale. Factored form (h@A@B) is implicit perfect cache. **Routing optimization DEAD:** router is 0.46% of inference (0.166ms/36ms), speculative selection gains <0.5% even at 100% accuracy. |
| Base-free | **25%** | Random scaffold killed. Ternary-from-scratch proven on toy task (MLX+STE), KILLED at d=512 on real text (PPL 2.78x, overfitting not mechanism failure — needs more data/regularization). GaLore+STE supported (0.28x optimizer state, fixes 2.6x degradation). **Warm-start scales to d=1024: 1.037x PPL ratio, LoRA adapters train stably on self-trained ternary base. Composition safe but vacuous (adapter deltas too small for meaningful test).** |
| **Overall** | **~54%** | Composition + routing validated (softmax router matches oracle at N=24, 0% fallback). E2E pipeline works (quality strong, latency open problem). Serving proven: 1.22 GB memory, 172 tok/s base, 97 tok/s adapter (addmm). Memory budget validated: 853 adapters fit on M5 Pro 48GB. Pre-merge latency proven at production scale. **Evolve: pruning validated** (LOO removes 20% with <1% impact). Base-free is gap. |

## Active Research Tracks (March 2026 Reframe)

### Track A: Own Our Ternary Base (P0)
- ~~Train ternary model from scratch on MLX using STE~~ **SUPPORTED** (PPL 1.003x FP32 on toy task, composition 1.022x, task overcapacity caveat). **KILLED at d=512 on real text** (PPL 2.78x, overfitting — needs more data + regularization, not mechanism failure)
- ~~GaLore+STE integration to fix 2-3x quantization degradation~~ **SUPPORTED** (PPL ratio 0.998x, optimizer state 0.28x, composition 1.019x, S2 FAIL at toy scale expected)
- Port MatMul-free LM to MLX — ternary + no matmul (exp_matmul_free_lm_mlx)
- Falcon-Edge onebitllms toolkit (tiiuae/onebitllms) as reference
- ~~Sparse-BitNet: exploit natural 42% sparsity~~ **KILLED** (sparse matmul 7% slower than packed BitLinear kernel. Inference bandwidth-bound; uint8 packing already optimal. Metal SIMD handles zeros for free.)
- GOAL: ternary base we control, trained on M5 Pro, supporting composition

### Track B: Smart Routing Without Heavy Infrastructure (P0)
- ~~Per-adapter tiny routing heads (~5K params each)~~ **SUPPORTED** (100% accuracy, 2.32% overhead, +19.9% over uniform, near-oracle 0.15% gap. Caveat: 5 trivially-separable domains only)
- ~~Hot-add adapter at runtime with zero retraining~~ **SUPPORTED** (K3: +0.70% avg degradation at N=5→6, 98.7% routing accuracy. Caveat: medical +3.5% from science-medical overlap)
- ~~Entropy-adaptive gating: skip experts when base is confident~~ **SUPPORTED** (63% tokens skip at 1.13% PPL cost, Otsu threshold CV=0.87/eta=0.68. Two-pass 2.1x slower — value is as pre-filter for routing heads, not standalone. 5 domains, uniform composition baseline)
- Test-Time Training for runtime expert selection (exp_ttt_expert_selection_mlx)
- ~~MoLoRA per-token routing on MLX~~ **SUPPORTED** (null result: per-token equivalent to per-sequence on clean domains, -0.46%. Gumbel-sigmoid mechanism works: 0.58% overhead, diversity 2.42. Needs mixed-domain data)
- ~~Text-to-LoRA hypernetwork for zero-training adapter generation~~ **KILLED** (K2 FAIL: convex-combination over 24 adapters + orthogonal projection = tautological kill. NN retrieval 1.28x PPL but redundant with softmax router. Need thousands of training pairs.)
- GOAL: efficient expert selection without dedicated router overhead

### Track C: Mechanism & Foundation (P1)
- Fix orthogonality.py to measure effective delta vec(B@A) (exp_bitnet_effective_delta_cosine)
- ~~XSA for composition quality improvement~~ **KILLED** (11.85% degradation, capacity cost outweighs at d_h=32)
- ~~AttnRes depth-wise attention for composition~~ **SUPPORTED** (K1 PASS, K3 PASS mechanism validated; K2 INCONCLUSIVE at L=4, needs L=16+ test)
- ~~Partial RoPE: position-free dims as routing features~~ **KILLED** (silhouette = -0.007, domain labels not the natural clustering axis for any internal representation. Full hidden states better than Q features but both have negative silhouette. Zero-parameter routing dead.)
- ~~PiSSA vs Grassmannian init~~ **KILLED** (PiSSA gives same A to all adapters → cos=1.0. Ternary SVD spectrum too flat. Grassmannian random orthonormal confirmed correct.)
- GOAL: settle the mechanism question, find better init strategies

### Track D: Production Serving on Apple Silicon (P2)
- Pre-merge is FREE (0.80% overhead proven on MLX)
- Per-token routing via pre-merge (merge selected experts before forward pass)
- ~~Sparse-BitNet for 1.5-2x speedup via natural ternary sparsity~~ **KILLED** (sparse ops slower on Apple Silicon, bandwidth-bound)
- GOAL: interactive serving on M5 Pro with dynamic expert selection

### Parked: Macro/CUDA Experiments (P5)
- All macro-scale RunPod experiments deprioritized
- Will revisit only after MLX-native mechanisms are proven
- Reasoning composition, 500-expert scaling, full base-free pipeline etc.

## Key References

### Ternary Architectures
- BitNet b1.58 (arxiv 2402.17764) — ternary architecture, the current base
- MatMul-free LM (arxiv 2406.02528) — ternary + no matmul, GRU attention, up to 2.7B
- Sparse-BitNet (arxiv 2603.05168) — natural 42% sparsity in ternary weights
- MoTE (arxiv 2506.14435) — frozen shared + ternary routed experts
- Falcon-Edge: tiiuae/onebitllms — open ternary training toolkit with Triton kernels

### Composition & Routing
- MoLoRA (arxiv 2603.15965) — per-token LoRA routing, 1.7B+4 adapters beats 8B
- Text-to-LoRA (arxiv 2506.06105) — hypernetwork generates adapters from text, ICML 2025
- PiSSA (arxiv 2404.02948) — SVD-init LoRA, NeurIPS 2024 spotlight
- Cross-LoRA (arxiv 2508.05232) — data-free LoRA transfer across base models
- LD-MoLE (arxiv 2509.25684) — learnable dynamic routing for MoLoRA experts
- Naive LoRA Summation (arxiv 2508.11985) — orthogonality enables additive composition
- LoRA Soups (arxiv 2410.13025) — CAT composition beats data mixing
- CLONE (arxiv 2506.02847) — MoE router for dynamic LoRA on edge devices

### Foundations
- GaLore (arxiv 2403.03507) — low-rank gradient training from scratch, <1% gap at 1B
- LoRI (arxiv 2504.07448) — frozen A + 90% sparse B, 17.3% better merge
- OSRM (arxiv 2505.22934) — data-aware orthogonality fixes weight-only gap, +12.78%
- FlyLoRA (arxiv 2510.08396) — frozen random A as implicit router, JL-lemma orthogonality
- KR-Test (arxiv 2601.03505) — knowledge retention eval via contrastive examples
- Cornerstone Layers (arxiv 2409.14381) — layer criticality explains base-free kill
- TTT Done Right (arxiv 2505.23884) — test-time training reference implementation

### Parameter Golf Intelligence
- https://github.com/openai/parameter-golf — $1M challenge, 16MB model in 10min
- TTT is biggest lever (#1: 1.1194 BPB via per-document adaptation)
- MoE fails below 500M params (Apple scaling laws, ICML 2025)
- N-gram + entropy mixing: 0.9674 BPB (15%+ over neural alone)
- XSA (Exclusive Self-Attention): zero-param attention fix, last 3-4 layers
- Partial RoPE (25% dims): position-free dims learn pure semantic similarity

### MLX / Apple Silicon
- MLX-BitNet: exo-explore/mlx-bitnet — first ternary impl for Apple Silicon
- M5 Neural Accelerators: up to 4x speedup over M4 for matmul-heavy workloads
- X-LoRA: EricLBuehler/xlora — mixture of LoRA experts, in HF PEFT
