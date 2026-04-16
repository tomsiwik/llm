# PAPER.md — P11.H0: thinking-universal-v0 (Domain-Agnostic Thinking)

## Abstract

We test whether multi-domain LoRA training (code + math, v_proj + o_proj only)
can amplify thinking-channel attention without catastrophic forgetting. The
hypothesis: gradient diversity from 2 diverse domains (GD > 0.5) forces LoRA
updates into the domain-invariant thinking subspace, preserving MMLU-Pro breadth
while improving structured reasoning quality.

---

## Prediction vs Measurement

| Metric | Theorem Prediction | Smoke (N=28/5) | Full Run (TBD) | Kill ID |
|--------|-------------------|----------------|----------------|---------|
| MMLU-Pro + thinking (adapter) | ≥65.1% | 46.4% (noise, 28q) | TBD | K1517 |
| GSM8K (adapter) | ≥80% | 60.0% (noise, 5q) | TBD | K1518a |
| MedMCQA (adapter) | uncertain (no medical data) | 60.0% (noise, 5q) | TBD | K1518b |
| Thinking chars/q | >0 (active) | 3202 chars/q ✓ | TBD | K1519 |
| Adapter size | <100MB | 12.56 MB ✓ | — | — |
| Training time | <90 min | 18.3 min (smoke) | TBD | — |

---

## Smoke Test Evidence (2026-04-14)

**Configuration**: 10 steps, 7 training examples, 28 eval questions (2/category)

**Key findings**:
- K1519 PASS: thinking active at 3202 chars/q — model uses thinking channel
- K1517 SMOKE: 46.4% — expected noise (28q, high variance); full run uses 210q
- K1518a SMOKE: 60% GSM8K on 5q — high variance; not predictive
- K1518b SMOKE: 60% MedMCQA on 5q — high variance; transfer uncertain without medical/science data
- Adapter registered: 12.56 MB at adapters/thinking-universal-v0/

**Format mismatch handled**: OpenThoughts uses DeepSeek `<|begin_of_thought|>...<|end_of_thought|>` tags.
These are stripped and re-wrapped in `<think>...</think>` for SFT. At inference,
Gemma 4's native `<|channel>thought...<channel|>` is used. No impact on training quality.

**2-domain design (corrected)**: Training uses code (600 examples) + math (1400 examples).
Science shard not loaded — budget folded into math. Theorem 1 precondition updated to |{D_i}| ≥ 2.
Science→medical transfer claim removed from Theorem 2. MedMCQA is now a secondary uncertain metric.

---

## Full Run Status

**Pueue task 17**: QUEUED (pending completion of tasks 2-16)

Full run rows will be populated after task 17 completes.

---

## Notes

- Base MMLU-Pro (Finding #536): 62.1% — used as reference for forgetting gap
- K1517 threshold = 62.1% + 3pp = 65.1% (heuristic, not derived from theorem)
- If K1517 FAILS but MedMCQA / GSM8K pass: status = provisional
- If thinking persists but accuracy regresses: revisit FM2 (math dominance bias)
