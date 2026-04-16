# LEARNINGS.md — P11.Z1: Injection Decoding

## Core Finding
Gemma 4 E4B generates a mean of 2614 thinking chars on MMLU-Pro questions — "Wait, Keep Thinking" injection decoding (arXiv:2503.10167) never triggered at 500-char threshold. The model does **not** exhibit premature thinking termination.

## Why
Injection decoding was designed for models that truncate reasoning early (e.g., s1 budget forcing in arXiv:2501.12599). Gemma 4 E4B's extended thinking mode consistently generates 1500+ chars even on straightforward questions. Threshold raised to 1500 before full run to get ~30-50% trigger rate.

## Implications for Next Experiment
- If K1533 (injection ≥ base + 1pp) FAILs at N=100: document as "Gemma 4 E4B does not under-think — injection decoding inapplicable". Pursue budget-reduction techniques instead (the problem is over-thinking, not under-thinking).
- If PS-prompt helps at N=100 despite instruction conflict: PS prefix works for Gemma 4 even with conflicting instructions.
- Key design lesson: verify trigger-rate assumptions with a smoke test before committing injection threshold. 500 chars was ~5× below the model's floor.

## Status
Full run (pueue task 13, N=100, 4 categories) still pending. Results will determine kill criteria K1532-K1534.
