# MATH.md — exp_hedgehog_domain_adapter_js

**Claim:** Per-layer cos-sim distillation from a JS-docs-in-context teacher into a rank-8 LoRA on Gemma 4 E4B produces a JavaScript domain adapter that beats base+token-space-LoRA on JS-specific tasks, with bounded non-interference on Python/natural language.

---

## 1. Failure mode

Degenerate: "JS nuance is captured only for specific syntactic patterns (e.g., arrow-function closures) and not for semantic rules (e.g., TDZ, hoisting). Held-out performance is pattern-bound." K2 held-out ≤ K2 train, K4 specificity passes weakly.

## 2. Cited prior math / findings

- Moudgil arxiv:2604.14191 §3.1 eq. 6.
- Pierre F#627 (r=6 LoRA captures domain specialization, Code +48pp).
- Pierre F#614 (thinking-mode required on reasoning tasks).
- HumanEval-JS as the JS-specific bench; HumanEval (Python) for non-interference.

## 3. Theorem (informal)

Identical structure to `exp_hedgehog_procedural_adapter_refactor` §3, with `π_R` replaced by `π_js` = JS documentation context and task replaced by JS-nuance Q-A.

**Theorem.** There exists rank-8 `Δθ` on `(v_proj, o_proj)` such that:
1. Per-layer cos > 0.80 on held-out JS prompts (K1)
2. JS-bench accuracy ≥ base + token-space LoRA at matched params (K2)
3. HumanEval (Python) drop < 3pp (K3)
4. MMLU subset drop < 2pp (K4)

**Proof sketch.** Same as refactor adapter. Domain adapters have strong empirical support on Gemma 4 E4B at r=6 (F#627).

## 4. Kill-criterion map

| KC | Measured quantity | Threshold | Type |
|---|---|---|---|
| K1 | mean per-layer cos on held-out JS Q-A | > 0.80 | structural proxy |
| K2 | JS-bench (HumanEval-JS + custom nuance suite) pass@1 vs base+token-LoRA | ≥ baseline | target (pair K1) |
| K3 | HumanEval (Python) drop vs base | < 3pp | target non-interference |
| K4 | MMLU subset drop vs base | < 2pp | target specificity |

## 5. Predicted measurements

- K1: mean cos ∈ [0.80, 0.87]
- K2: ΔJS-bench ∈ [0, +3pp] vs token-space LoRA
- K3: HumanEval drop ≤ 2pp
- K4: MMLU drop ≤ 1pp
