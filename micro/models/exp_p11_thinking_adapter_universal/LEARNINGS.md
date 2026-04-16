# LEARNINGS.md — P11.H0: thinking-universal-v0

## Core Finding

2-domain LoRA (code+math, 2000 examples, v_proj+o_proj r=8) successfully activates
thinking channels (3202 chars/q in smoke) without collapsing to a narrow domain.
Full-run results pending (pueue task 17).

## Why

Gradient diversity from code + math domains forces LoRA updates into the
domain-invariant thinking subspace (arXiv:2310.02207 gradient diversity framing).
Thinking channel activation confirmed even at 10 training steps — the channel is
structurally receptive to LoRA stimulation.

## Key Caveats (from review)

- Theorem 1 Cauchy-Schwarz → FG scaling step is asserted, not derived; acceptable
  for guided exploration.
- K1517 (+3pp over 62.1% baseline) is a heuristic threshold, not theorem-derived.
- MedMCQA (K1518b) is explicitly uncertain: no medical/science data in training.
- Format mismatch: OpenThoughts uses DeepSeek tags; stripped and re-wrapped in
  `<think>...</think>` for SFT; native Gemma 4 tags used at inference.

## Implications for Next Experiment

If full run K1517 FAILS (MMLU-Pro < 65.1%) but GSM8K passes: the thinking
subspace is domain-specific, not universal — next step is to test whether
adding a 3rd domain (science or medical QA) closes the forgetting gap.
If K1517 PASSES: universal thinking adapter is feasible; scale to 5-domain
training with N_TRAIN = 5000 and test composition with math-s1k-reasoning-v0.
