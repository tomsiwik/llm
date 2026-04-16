# LEARNINGS: exp_p11_baseline_eval (P11.E0)

**Status: QUEUED — PAPER.md pending results**

## Core Finding (Design Rationale)
Adapter evaluations are reusable: Score(θ_base + Δ_k, S_B) is independent of all other adapters and remains valid until Δ_k is retrained (Theorem 1, load-one-at-a-time evaluation). This means a single baseline sweep fills all gaps in the adapter registry permanently.

## Why This Experiment Exists
The adapter registry had eval scores only for math-gsm8k-knowledge-v0 (GSM8K=82%), with thinking-mode evals and most adapter × benchmark combinations missing. Without these baselines, comparing reasoning adapters (P11.A0/A1) against knowledge adapters is impossible. The registry blind spot was identified as a prerequisite blocker.

## Design Notes
- Evaluates 5 adapters × 2 benchmarks × 2 thinking modes = ~20 eval runs
- Thinking mode creates distribution mismatch for non-thinking-trained adapters (Finding #536): knowledge adapters expected to show partial thinking suppression
- GSM8K only evaluated with thinking=ON (by design, per MATH.md)
- Per-adapter load/unload with mx.clear_cache() prevents memory overflow

## Key Fix Applied (Adversarial Review)
`data/` directory not created before GSM8K write — fixed with `mkdir(parents=True, exist_ok=True)`. No logic changes.

## Predictions (to verify in PAPER.md when results arrive)
- Base MMLU-Pro (thinking=ON): ~62.1% (Finding #530 validation)
- Base MMLU-Pro (thinking=OFF): ~40%
- Knowledge adapters: thinking chars/q < base (partial suppression from Finding #536)
- Math adapter MMLU-Pro: above base (domain lift expected)

## Implications for Next Experiment
Once results arrive and registry.json is updated, P11.A0 (s1K SFT) and P11.A1 (LIMO SFT) can be compared against these baselines. The thinking suppression delta (base vs. knowledge adapter) will establish whether reasoning-trained adapters successfully recover thinking capability.
