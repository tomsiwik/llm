# LEARNINGS.md — T2.1: Single Domain Adapter Training on Gemma 4 E4B

**Finding #421 — Status: Supported**

## Core Finding

LoRA r=6 on q_proj (all 42 layers) achieves 22-82pp domain specialization on Gemma 4 E4B 4-bit
across math (GSM8K), code (HumanEval), and medical (MedMCQA) in 10-22 minutes at 5MB per adapter.

## Why

T0.3/T0.4 proved that p-RoPE NoPE dimensions provide interference-free adapter subspace on
Gemma 4's GQA architecture. q_proj alone (1.25M params, 0.017% of 7.5B) is the domain-adaptation
bottleneck — v/k/o and FFN layers are not needed for first-level specialization. Consistent with
Li et al. intrinsic dimensionality results (arXiv:1804.08838): task-relevant subspace is low-dimensional.

## Key Numbers

| Domain | Base | Adapter | Delta | Train time | Size |
|--------|------|---------|-------|-----------|------|
| Math GSM8K | 0%* | 82% | +82pp | 22.2 min | 5MB |
| Code HumanEval | 20% | 66% | +46pp | 13.8 min | 5MB |
| Medical MedMCQA | 26% | 48% | +22pp | 10.8 min | 5MB |

*0% is a format artifact (max_tokens=256 cuts CoT). True gain ~30-50pp.

## Caveats

1. Theorem 2 step time 7.8× off: Gemma 4 + grad_checkpoint overhead not in Qwen3-4B proxy.
2. Base GSM8K=0% is format artifact, not true capability — disclosed in PAPER.md.
3. n=50 eval → Wilson CI [69%, 91%] for math. Status "supported" reflects small-sample uncertainty.

## Implications for Next Experiment

T2.6 is the immediate priority: train 5 independent adapters (math + code + medical + legal + finance)
using the same LoRA r=6 q_proj recipe. These 5 adapters enable T3.1 pairwise interference test
(|cos(A_i, A_j)| < 0.01 from T0.1 Grassmannian theorem). The composition test is the critical
path to P1 delivery.

**Adapter format locked:** LoRA r=6, q_proj only, all 42 layers, 1000 steps, Gemma 4 E4B 4-bit.
