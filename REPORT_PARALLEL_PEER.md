# Parallel PEER Branch Benchmark Report

**SmolLM-135M + PEER Experts, 3-Domain Sequential Learning (wiki → code → math)**
**200 training steps/domain, 500 train seqs, 50 eval texts per domain**

---

## Hypothesis

Branch-level isolation (frozen branches have zero routing interference with active branches) beats neuron-level lifecycle (frozen and active experts share the same routing space) for catastrophic forgetting reduction.

## Configurations

| Config | Architecture | Experts | Active/Token | Stored Params | s/step |
|--------|-------------|---------|-------------|--------------|--------|
| `peer_lifecycle` | 1 PEER layer, freeze top-64 experts | 1024/layer | 32 | 80.1M | 0.65 |
| `peer_parallel_2b` | 2 competing branches, softmax gate | 529/branch | 17/branch | 102.5M | 0.82 |
| `peer_parallel_2b_gpu` | same + 2 shared GPU streams | 529/branch | 17/branch | 102.5M | 0.92 |

**Lifecycle strategy:**
- `peer_lifecycle`: After each domain, freeze top-64 experts per layer + recycle dead
- `peer_parallel_2b*`: After wiki, freeze best branch entirely. After code, intra-branch top-32 expert freeze. Math trains on remaining active branch.

---

## Results

### Perplexity Matrix (lower = better)

**Baseline (frozen SmolLM-135M, no adapters):** wiki=32.1, code=5.9, math=6.8

| Config | After | wiki | code | math |
|--------|-------|-----:|-----:|-----:|
| `peer_lifecycle` | T0(wiki) | 35.1 | 9.6 | 11.9 |
| | T1(code) | 48.7 | 8.4 | 8.9 |
| | **T2(math)** | **61.3** | **11.6** | **4.7** |
| `peer_parallel_2b` | T0(wiki) | 40.9 | 9.9 | 13.2 |
| | T1(code) | 52.3 | 8.5 | 9.5 |
| | **T2(math)** | **79.9** | **11.1** | **5.2** |
| `peer_parallel_2b_gpu` | T0(wiki) | 38.6 | 9.6 | 13.8 |
| | T1(code) | 49.5 | 8.8 | 9.5 |
| | **T2(math)** | **64.9** | **11.8** | **4.6** |

### Forgetting (% perplexity increase from best → final)

| Transition | peer_lifecycle | parallel_2b | parallel_gpu | Best |
|-----------|---:|---:|---:|---|
| T1>wiki (code overwrites wiki) | +38.9% | **+28.0%** | +28.1% | parallel |
| T2>wiki (math overwrites wiki) | +74.8% | +95.2% | **+67.9%** | parallel_gpu |
| T2>code (math overwrites code) | +38.3% | **+30.6%** | +33.9% | parallel |
| **Mean Forgetting** | **+50.7%** | **+51.3%** | **+43.3%** | **parallel_gpu** |

### Wall-Clock Performance

| Config | Time/Step | Total (3 domains) | Overhead vs baseline |
|--------|----------|-------------------|---------------------|
| `peer_lifecycle` | 0.65 s | 388 s | — |
| `peer_parallel_2b` | 0.82 s | 496 s | +28% |
| `peer_parallel_2b_gpu` | 0.92 s | 554 s | +42% |

---

## Key Findings

### 1. GPU streams with shared streams: best overall forgetting

**`peer_parallel_2b_gpu` achieves 43.3% mean forgetting — the lowest across all configs** (vs 50.7% lifecycle, 51.3% no-stream parallel). The two shared GPU streams enable better concurrent scheduling of the computation graph across branches.

Surprisingly, GPU streams don't just affect latency — they change the training dynamics. The branch that gets frozen (and which branch that is) differs between runs due to random initialization, but the stream version consistently produces lower T2>wiki forgetting (67.9% vs 95.2% without streams).

### 2. Branch isolation wins the first domain transition

Both parallel configs reduce T1>wiki forgetting from +38.9% (lifecycle) to +28% — a **28% relative reduction**. When wiki branch is frozen and code trains on the other branch, there's complete routing isolation: no shared expert pool, no routing competition.

### 3. Gate doesn't specialize — shared trainable bottleneck

Gate weights at evaluation time:
```
parallel_2b:     wiki: B0=0.54, B1=0.46  |  code: same  |  math: same
parallel_gpu:    wiki: B0=0.42, B1=0.58  |  code: same  |  math: same
```

The gate learns a **static bias** (slight preference for one branch), not domain-dependent routing. All three domains get the same gate weights. This means later domain training can "mute" the frozen branch by shifting gate weights, causing indirect forgetting.

This is the same failure mode as LoRA self-routing (all atoms identical) and SwitchRouter (4.3% accuracy). **Learned input-dependent routing consistently fails to specialize in our setup.**

### 4. No-stream parallel: worse than lifecycle on T2>wiki

Without streams, `peer_parallel_2b` has 95.2% wiki forgetting after math — worse than lifecycle's 74.8%. The gate drift problem is severe: math training shifts gate weights away from the frozen wiki branch, effectively muting it despite perfect weight isolation.

GPU streams mitigate this (67.9%), possibly because the different scheduling order changes gradient flow through the gate.

### 5. Parameter budget: +28% for parallel

Parallel configs use 102.5M stored params vs 80.1M for lifecycle (+28%). The bulk comes from duplicating `query_proj` (2 × d_key × d_in = 331K) per branch. The expert weights themselves are roughly budget-matched (2 × 529 ≈ 1058 vs 1024).

### 6. MLX stream architecture lesson

**Per-layer streams crash Metal.** Creating `mx.new_stream()` per `ParallelPEERLayer` (60 layers × 2 branches = 120 Metal command queues) causes the ~5s GPU watchdog timeout. The fix: create 2 streams globally via `setup_parallel_streams()` and share them across all 60 layers.

**Heterogeneous GPU/CPU streams are impractical** for matmul-heavy workloads — 80x slower in microbenchmarks because CPU has no Metal accelerator for matrix operations.

---

## Does Parallel Beat Sequential?

**Mixed verdict:**

| Metric | Winner | Margin |
|--------|--------|--------|
| Mean forgetting | **parallel_gpu** (43.3%) | -15% vs lifecycle (50.7%) |
| First-domain preservation (T1>wiki) | **parallel** (28.0%) | -28% relative vs lifecycle (38.9%) |
| Late-domain preservation (T2>wiki) | **parallel_gpu** (67.9%) | -9% vs lifecycle (74.8%) |
| Code forgetting (T2>code) | **parallel** (30.6%) | -20% vs lifecycle (38.3%) |
| Final wiki ppl | **lifecycle** (61.3) | vs parallel_gpu 64.9 |
| Final math ppl | **parallel_gpu** (4.6) | vs lifecycle 4.7 |
| Latency | **lifecycle** (0.65 s/step) | vs parallel_gpu 0.92 (+42%) |
| Param budget | **lifecycle** (80.1M) | vs parallel 102.5M (+28%) |

**Parallel with GPU streams is the best forgetting reducer**, but at the cost of +42% latency and +28% params. The benefit comes primarily from branch-level isolation protecting the first domain.

The **no-stream parallel** variant is not recommended — it's slower than lifecycle AND has worse T2>wiki forgetting due to gate drift.

---

## Architectural Weaknesses

1. **Gate doesn't specialize.** The softmax gate over branches learns a static bias, not per-domain routing. Frozen branches get muted over time as the gate shifts during later training. Potential fix: freeze gate contributions for frozen branches (hard-wire their weight).

2. **2 branches < 3 domains.** Domains 2 and 3 share one branch, causing intra-branch forgetting. With 3 branches (324=18² experts each ≈ 972 total), each domain gets its own branch.

3. **Parameter overhead.** Per-branch routing infrastructure (query_proj, sub_keys) is duplicated. Sharing query_proj across branches or using smaller d_key could reduce the +28% overhead.

---

## Recommendations

1. **Use `peer_parallel_2b_gpu` if forgetting reduction is priority** and the +42% latency is acceptable.
2. **Stick with `peer_lifecycle` if latency/budget matters** — simpler, cheaper, comparable forgetting.
3. **Next experiment: freeze gate for frozen branches** — hard-wire frozen branch gate weight to its value at freeze time. This directly addresses the gate drift problem.
4. **Next experiment: 3 branches** (324 experts each) — one per domain, eliminates shared-branch forgetting.
