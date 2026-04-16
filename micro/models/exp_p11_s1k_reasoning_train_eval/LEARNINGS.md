# LEARNINGS.md — P11.F0 exp_p11_s1k_reasoning_train_eval

**Status:** Queued (pueue task 12) — pre-run design context

## Core Finding (Design)
P11.A0 catastrophic forgetting (-26pp MMLU-Pro) was caused by extreme overfitting to 27
examples (37 epochs), not inherent domain incompatibility of competition math. Increasing
MAX_TOTAL_CHARS from 6000 → 32000 recovers 831 training examples and reduces to ~1.2 epochs.

## Why
Theorem 1 (epoch-count drift bound): KL divergence from base scales linearly with N_epochs.
With 31× fewer epochs (37 → 1.2), expected forgetting drops from -26pp to ~-0.8pp.
Citation: Li & Liang 2021 + LoRA paper (arXiv:2106.09685) on catastrophic forgetting.

## Key Fix Applied (from REVISE)
K1508 threshold corrected: 65% → 59% (≥ base − 3pp = 62.1 − 3.1pp).
Theorem 1 predicts ~61-63% MMLU-Pro; original 65% threshold would have marked a confirmed
result as FAIL.

## Implications for Next Experiment
- If K1508 PASS (≥59%): epoch theory confirmed → s1K is viable, domain mismatch was NOT
  the root cause. Next: LIMO (capability-boundary gradient theory, exp_p11_reasoning_sft_limo).
- If K1508 FAIL (<59%): domain mismatch theory holds regardless of overfitting fix →
  need multi-subject reasoning traces (Sky-T1, MMLU-Pro formatted) instead of competition math.
- Watch avg_thinking_chars: if 0 post-training, thinking format mismatch is a confound
  (training used `<think>` tags, Gemma 4 uses `<|channel>thought...<channel|>`).

## Non-Blocking Risks to Document in PAPER.md
1. Thinking training format mismatch: `<think>` in data ≠ Gemma 4 native `<|channel>` format
2. GSM8K N=50 is noisy: SE ≈ 5.6pp, 80% threshold unreliable at this sample size
3. Phase 4a base eval redundant: base = 62.1% already established (Finding #530)
