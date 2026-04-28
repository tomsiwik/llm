# Adversarial Review — exp_p1_cross_domain_interference

**Reviewer verdict: KILL (clean)**

## Checklist

| # | Check | Result |
|---|-------|--------|
| (a) | results.json verdict matches DB status | PASS — both KILLED |
| (b) | all_pass matches claim | PASS — false, K2067 FAIL |
| (c) | PAPER.md verdict matches DB status | PASS — KILLED |
| (d) | is_smoke → provisional | N/A — is_smoke=false |
| (e) | KC not modified after first run | N/A — untracked, single run |
| (f) | No tautological KC | PASS — both are task accuracy thresholds |
| (g) | Code measures what MATH.md describes | PASS — off-domain delta from base, on-domain delta from base |
| (h) | No independent lora_A/lora_B summation | PASS — no composition, individual adapter loading |
| (i) | LORA_SCALE < 12 | N/A — no composition |
| (j) | No single-sample routing applied to all | N/A — no routing |
| (k) | No shutil.copy fake adapters | PASS |
| (l) | No hardcoded pass: True | PASS |
| (m) | Model in MATH.md = model in code | PASS — gemma-4-e4b-it-4bit throughout |
| (n) | Base accuracy > 0% | PASS — 50%/22%/6% |
| (o) | N ≥ 15 | PASS — N=50 |
| (p) | Target-metric KC present | PASS — both KCs are task accuracy |

## Notes
- Clean experimental design: 3×3 matrix with base-model control, no composition complexity.
- K2067 FAIL is clear: python→GSM8K -14pp, medical→HumanEval -12pp.
- K2068 PASS confirms adapters are strong on-domain (+22 to +62pp).
- Finding #827 correctly identifies asymmetric interference and need for soft routing.
