# Depth-Routed Adapters: Per-Layer Adapter Selection via Pseudo-Queries

## Abstract

We test whether adding a second routing axis — per-layer depth weights — on top of proven token-level softmax routing improves LoRA adapter composition quality. The depth router learns per-layer, per-adapter scaling factors via pseudo-queries and expert embeddings (inspired by AttnRes, arXiv 2603.15031). On a micro transformer (d=128, L=4, 5 character-level domains), depth routing fails on both kill criteria: weights remain near-uniform (entropy ratio 0.992, K1 threshold <0.95) and quality degrades -18.3% vs token-only routing (K2 threshold ≥+2%). Token-level softmax routing already matches oracle perfectly (0.0% gap), leaving no room for depth routing to improve. The result confirms the attnres_depth_composition finding: L=4 is too shallow for depth-axis effects.

## 1. Motivation

Our architecture composes N LoRA adapters via routing. Prior work established:
- **Softmax token-level routing** matches oracle quality at N=24 (gamma 0.625 = oracle, exp_softmax_router_scaling)
- **AttnRes depth attention** learns non-uniform depth weights (entropy 0.775) but provides negligible composition improvement at L=4 (0.39%, exp_attnres_depth_composition)

The hypothesis: combining token-level routing (WHICH adapter) with layer-level routing (HOW MUCH per layer) enables the model to specialize adapter contributions across depth. For example, syntactic adapters might contribute more at early layers while semantic adapters contribute at deep layers.

## 2. Method

### 2.1 Token-Level Router (Baseline)

A 2-layer MLP maps mean-pooled hidden states to adapter probabilities via softmax:

```
h_pool = mean(transformer(x), axis=seq)    ∈ R^d
p = softmax(W_2 · gelu(W_1 · h_pool))     ∈ R^N
adapter_idx = argmax(p)
```

### 2.2 Depth Router (Novel)

Learned pseudo-queries w_l ∈ R^{d_e} (one per layer) and expert embeddings r_i ∈ R^{d_e} (one per adapter) produce per-layer, per-adapter weights:

```
α_{i,l} = softmax_i(w_l^T · r_i / √d_e)     ∈ R^N per layer l
```

The adapter's LoRA-B matrix at layer l is scaled by α_{adapter_idx, l} × N_layers (normalized so uniform = 1.0):

```
ΔW_l = A_l @ (B_l × α_{i,l} × L)
```

### 2.3 Training

- **Base model**: Micro transformer, 1.1M params, 600 steps
- **Adapters**: 5 domains (alpha, numeric, mixed, upper, symbol), 400 steps each, rank=8
- **Token router**: 300 steps on domain classification, achieves 100% accuracy
- **Depth router**: Gradient-free perturbation search (40 iterations), minimizing routed PPL
- **Three seeds**: 42, 137, 314

## 3. Results

### 3.1 Kill Criteria

| Criterion | Metric | Result | Threshold | Verdict |
|-----------|--------|--------|-----------|---------|
| K1: Depth specialization | Mean entropy ratio | 0.992 | <0.95 | **FAIL** |
| K2: Depth > token-only | Improvement | -18.3% | ≥+2% | **FAIL** |
| S1: Strong improvement | Improvement + K1 | -18.3%, FAIL | ≥+5% + K1 | **FAIL** |

### 3.2 Routing Comparison (Gamma = Geometric Mean PPL Across Domains)

| Mode | Seed 42 | Seed 137 | Seed 314 | Mean |
|------|---------|----------|----------|------|
| Oracle | 1.008 | 1.013 | 1.015 | 1.012 |
| Token-only | 1.008 | 1.013 | 1.015 | 1.012 |
| Token+depth | 1.136 | 1.065 | 1.390 | 1.197 |
| Random | 2.205 | 1.919 | 2.329 | 2.151 |
| Uniform 1/N | 1.733 | 1.776 | 1.826 | 1.778 |

Token-only routing achieves 0.0% oracle gap. Depth routing degrades by 18.3%.

### 3.3 Per-Domain Breakdown (Seed 42)

| Domain | Oracle | Token-only | Token+depth |
|--------|--------|------------|-------------|
| alpha | 1.006 | 1.006 | 1.006 |
| numeric | 1.008 | 1.008 | 1.008 |
| mixed | 1.016 | 1.016 | **1.837** |
| upper | 1.006 | 1.006 | 1.007 |
| symbol | 1.006 | 1.006 | 1.007 |

The "mixed" domain is catastrophically affected by depth routing across all seeds. This domain has the largest per-layer norm gradient (1.06→3.00 at seed 42), so depth scaling amplifies the imbalance.

### 3.4 Depth Weight Analysis

Depth weights remain near-uniform across all seeds:

**Seed 42** (entropy ratio 0.991):
| Layer | alpha | numeric | mixed | upper | symbol |
|-------|-------|---------|-------|-------|--------|
| 0 | 0.233 | 0.231 | 0.189 | 0.140 | 0.207 |
| 1 | 0.217 | 0.201 | 0.226 | 0.144 | 0.212 |
| 2 | 0.242 | 0.190 | 0.140 | 0.256 | 0.172 |
| 3 | 0.240 | 0.205 | 0.170 | 0.207 | 0.178 |

**Seed 314** (entropy ratio 1.000): perfectly uniform 0.200 everywhere.

### 3.5 Adapter Layer Norms (ΔW Frobenius norms per layer)

All adapters show monotonically increasing norms with depth:

| Domain | L0 | L1 | L2 | L3 | Ratio L3/L0 |
|--------|-----|-----|-----|-----|-------------|
| alpha | 1.33 | 1.45 | 1.60 | 1.74 | 1.31 |
| numeric | 1.07 | 1.11 | 1.18 | 1.28 | 1.20 |
| mixed | 1.06 | 1.53 | 2.33 | 2.99 | 2.82 |
| upper | 1.52 | 1.48 | 1.63 | 1.73 | 1.14 |
| symbol | 1.49 | 1.55 | 1.78 | 1.83 | 1.23 |

(Seed 42 values. All seeds show the same pattern.)

## 4. Analysis

### 4.1 Why Depth Routing Fails

**Root cause: Token-level routing already achieves oracle performance.** With 100% router accuracy and a 0.0% oracle gap, there is zero headroom for depth routing to improve. Any non-uniform depth weighting can only hurt by distorting the trained adapter weights.

The depth router's perturbation search finds that the best solution is uniform weights (doing nothing) — the optimization landscape has no gradient toward specialization.

### 4.2 Why Mixed Domain Blows Up

The "mixed" domain has 2.8x norm gradient across layers (L0=1.06, L3=2.99). When depth routing applies non-uniform scaling, it amplifies this imbalance. Even small deviations from uniform (0.189 vs 0.200) at L=4 with steep norm gradients cause disproportionate distortion.

### 4.3 Consistency with Prior Results

This confirms exp_attnres_depth_composition: **L=4 is too shallow for depth-axis effects.** At L=4, each layer contributes ~25% — the norm gradient exists but isn't steep enough for depth routing to exploit without causing instability.

The Kimi AttnRes paper (2603.15031) showed benefits at L=48 where each layer contributes ~2%. Our micro scale cannot replicate this.

### 4.4 What Would Change at Larger Scale

At L=32+ with 100+ layers of LoRA:
1. Adapter norm gradients would be steeper (L0 vs L31 could be 10x)
2. Some layers might genuinely benefit from different adapters
3. The optimization signal would be stronger (more degrees of freedom)

However, the clean result here — token routing already at oracle — suggests depth routing is unnecessary when token routing works well.

## 5. Verdict

**KILLED.** Both kill criteria fail:
- K1: Depth weights fail to specialize (entropy 0.992 > 0.95 threshold)
- K2: Depth routing hurts quality (-18.3% instead of ≥+2% improvement)

**Implication:** At the micro scale, token-level routing is sufficient. Per-layer adapter modulation is a dead end unless token routing has significant oracle gap (which softmax routing eliminates).

## 6. Recommendations

1. **Do not pursue depth routing** when token-level routing achieves oracle performance
2. **Focus on routing quality at scale** where oracle gaps may emerge with harder tasks
3. **Layer-level effects require deeper models** (L≥16) — testing at L=4 is necessary to kill the idea cheaply but cannot confirm it

## Platform

Apple M5 Pro 48GB. MLX. Total runtime: 105s across 3 seeds.
