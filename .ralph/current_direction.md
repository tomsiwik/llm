# Current Direction: P11.G0 GRPO Improve — Awaiting Reviewer

## Orphan Fix (this iteration)
- exp_p11_grpo_improve: PAPER.md written, emitting experiment.done → Reviewer
- exp_p11_thinkpo_polish: still needs PAPER.md + experiment.done (next iteration)

## Previous State: P11 Experiments Running

## Pueue Queue Status
- Task 0: exp_p11_reasoning_sft_s1k (RUNNING since Apr 13 23:12 — base eval MMLU-Pro thinking)
- Task 1: exp_p11_baseline_eval (QUEUED — will run after s1K)
- Task 2: exp_p11_plan_and_solve_prompt (QUEUED — will run after baseline_eval)

## Pending PAPER.md writes
When each task completes, read results.json and write PAPER.md + complete in DB:
- s1K: MATH.md exists, kill criteria K1515-K1518
- baseline_eval: MATH.md + LEARNINGS.md exist, kill criteria K1505-K1507
- plan_and_solve: MATH.md just written, kill criteria K1529-K1531

## plan_and_solve Kill Criteria
- K1529: best prompt + thinking >= 64% MMLU-Pro (>= 2pp over 62.1% baseline)
- K1530: PS+ >= PS accuracy (self-check adds value)
- K1531: best prompt output token count <= 2x direct-answer count

## baseline_eval Kill Criteria
- K1505: All 5 adapters evaluated on GSM8K + MMLU-Pro (thinking ON and OFF)
- K1506: Base model MMLU-Pro+thinking ≈ 62.1% (Finding #530 validation)
- K1507: registry.json updated with all eval scores

## exp_p11_w4a16_verification (pueue task 4, QUEUED)
- MATH.md + run_experiment.py written, LEARNINGS.md written
- Kill: K1538 (confirm W4A16), K1539 (N/A if W4A16), K1540 (8-bit >= 4-bit + 5pp)

## exp_p11_reasoning_sft_limo (pueue task 5, QUEUED)
- MATH.md written: capability-boundary gradient maximization theorem (arXiv:2502.03387)
- run_experiment.py written: GAIR/LIMO dataset, correct Gemma 4 thinking regex
- Kill: K1493 (≥65% MMLU-Pro), K1494 (≥85% GSM8K), K1495 (<1h training)

## CRITICAL BUG IN s1K (task 0)
- Phase 4a base eval: 12.5% / 0 thinking chars → INVALID
- Cause: strip_thinking uses <think>...</think> but Gemma 4 uses <|channel>thought...<channel|>
- Phase 4b adapter: same bug but results may be partially valid
- PAPER.md should document this and note re-run needed

## exp_p8_xgrammar_constrained_generation (pueue task 7, QUEUED)
- MATH.md + run_experiment.py written: self-repair grammar constraint protocol
- Tests: direct vs think-then-code vs self-repair on 20 Python problems
- Kill: K1333 (≤2% syntax errors after N=3 retries), K1334 (<5pp accuracy drop), K1335 (<5% overhead)
- cite: arXiv:2411.15100 (XGrammar), arXiv:2601.07525 (Think-then-constrain)

## Next unclaimed work
- exp_p11_grpo_reasoning (open) → claim after LIMO design complete

## Key path fix
run_experiment.py files need REPO_ROOT = Path(__file__).parent.parent.parent (3 levels up from micro/models/exp_name/)
