# REVIEW — exp_p1_n3_composition_test

**Verdict: KILL**

## Adversarial Checklist

| Check | Result |
|-------|--------|
| (a) results.json verdict matches DB | PASS — both KILLED |
| (b) all_pass matches claim | PASS — false |
| (c) PAPER.md verdict matches | PASS — KILLED |
| (d) is_smoke vs status | PASS — is_smoke=false, verdict=KILLED |
| (e) KC not modified post-run | PASS — KCs in MATH.md match code |
| (f) No tautological KC | PASS — K2062/K2064 are real benchmarks |
| (g) Code measures what MATH.md says | PASS — Σ(A_i @ B_i) with scale/N |
| (h) Composition math correct | PASS — line 105: per-adapter `a_i @ b_i` then sum |
| (i) LORA_SCALE safe | PASS — 6.0/3 = 2.0 per adapter |
| (j) No single-sample routing leak | N/A — uniform composition |
| (k) No shutil.copy adapter faking | PASS |
| (l) No hardcoded pass | PASS |
| (m) Model consistent | PASS — gemma-4-e4b-it-4bit throughout |
| (n) Base accuracy check | PASS — nonzero baselines |
| (o) N ≥ 15 | PASS — N=50 |
| (p) Target-metric KC present | PASS — K2062, K2064 |

## Assessment

Clean kill. K2062 fails definitively: math -10pp, code -12pp under uniform Σ(A_i@B_i)/N. The experiment correctly demonstrates that composition is non-tautological (PPL changes) and cross-domain interference is low (14% math→MedQA), but uniform scale allocation is insufficient. This directly motivates per-sample routing as the next step.
