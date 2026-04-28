# Adversarial Review — exp_p1_n3_routing_accuracy

## Checklist

**Consistency:**
- (a) results.json verdict = KILLED, DB status = killed — MATCH ✓
- (b) results.json all_pass = false — MATCH ✓
- (c) PAPER.md verdict = KILLED — MATCH ✓
- (d) is_smoke: false — N/A ✓

**KC integrity:**
- (e) KC not modified after first run — PASS (single run, no iterations) ✓
- (f) No tautological KC — PASS. K2065 tests router accuracy (could fail with random features). K2066 tests behavioral routed vs uniform (the hypothesis under test) ✓
- (g) Code measures same quantity as MATH.md — PASS. MATH.md predicts router ≥85% and routed > uniform; code measures exactly these ✓

**Code bugs:**
- (h) Composition math: line 167 `w[a_key] @ w[b_key]` = A @ B per-adapter, line 169 summed and scaled. Correct Σ(A_i @ B_i) ✓
- (i) LORA_SCALE = 6.0 — safe ✓
- (j) Per-domain routing shortcut: code applies one adapter per benchmark rather than per-sample. Valid because each benchmark is single-domain, so a perfect router would pick the same adapter for all samples within a benchmark. Not a bug ✓
- (k) No shutil.copy ✓
- (l) No hardcoded pass:true ✓
- (m) Model = gemma-4-e4b-it-4bit in code, PAPER.md, and adapter configs — consistent ✓

**Non-blocking flags:**
- (n) No base accuracy measured (not needed for this experiment) ✓
- (o) N=30 — marginal but delta is consistent across all 3 domains (routed ≤ uniform everywhere). Not noise ✓
- (p) K2066 is target-metric (behavioral task accuracy) ✓

## Concerns

1. **N=30 vs N=50 discrepancy with F#825**: Composition test (N=50) showed uniform worse than single-adapter. This test (N=30) shows uniform better than single-adapter routing. Sample variance or eval subset differences may explain this. The key takeaway (hard routing doesn't help) stands regardless.

2. **Scale factor untested**: Both conditions use scale=6.0. Different scales could shift the comparison. Acceptable for a kill — the hypothesis was about routing, not scale tuning.

## Verdict: KILLED

Clean kill. Router is trivially solved (100%) but the hypothesis that per-sample hard routing beats uniform composition is falsified. Finding #826 logged.
