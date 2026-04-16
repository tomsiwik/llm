# LEARNINGS: exp_p11_reasoning_sft_s1k

## Core Finding
Competition math SFT (s1K, 1000 steps, 27 examples) caused catastrophic forgetting on MMLU-Pro:
adapter dropped from 62.1% base to 36.1% (−26pp). Thinking was preserved (1641 chars/q).

## Why
s1K traces are near-orthogonal to MMLU-Pro's token distribution in embedding space. Gradient
descent on math-only traces pushes the model away from general reasoning breadth. This is not
a hyperparameter problem — it's a distributional mismatch at the trace level (arXiv:2502.03387
shows LIMO quality > quantity, but subject diversity is not covered). The math category itself
only scored 20%, ruling out "domain gain + forgetting" — it's pure degradation.

## Data Correction (from adversarial review)
Per-category table in PAPER.md has two wrong rows (base eval bug contaminated Phase 4a):
- biology: actual adapted = ~50%, not 10%
- computer science: actual adapted = ~40%, not 5%
K1491 should be labeled INVALID (HTTP 422 = untestable), not FAIL.

## Implications for LIMO (P11.A1, pueue task 5)
GAIR/LIMO is harder olympiad competition math — expect similar or worse degradation.
If math category scores <25% with LIMO adapter, kill immediately without full eval.
LIMO's training format uses `<think>...</think>` which may not match Gemma 4's actual
`<|channel>thought...<channel|>` tokens — check avg_thinking_chars in Phase 3 eval.

## Impossibility Structure
SFT preserves MMLU-Pro accuracy ONLY IF training traces span MMLU-Pro's 14-category
token distribution. Single-domain traces (math only) are structurally insufficient.
Required fix: diverse traces across all MMLU-Pro categories, not math-only datasets.
