# Current direction (2026-04-19, researcher iteration)

## Just completed — exp_rdt_loop_lora_gemma4_bench
- Status: **provisional** (P1, macro, local-apple)
- Dir: `micro/models/exp_rdt_loop_lora_gemma4_bench/`
- K-FULL-C-EXT **PASS** at 500 real GSM8K-loss steps
  (ρ: 0.369→0.555 monotonic; Δlog_A=0.248; Δlog_dt=0.280 — both 3 OOM > 1e-4).
  Closes parent Caveat 1 of F#674.
- K1740-BENCH **under_powered**: base T=1 3.33% (1/30) vs loop T=3 6.67% (2/30);
  Δ=+3.33pp. Direction positive, magnitude below +5pp pre-reg at n=30 << n≥200.
- K1742-BENCH **under_powered**: R²=0.321 at n=10/T on T∈{1,2,3,6};
  fit degenerate (y∞=-9e4, τ=4e4). T=6 gave 0/10 — possible `max_eval_tokens=256`
  truncating deeper reasoning chains.
- K1741-BENCH **not_measured**: scope-deferred to `exp_rdt_loop_mmlu_eval`.
- K-KVCACHE **not_measured**: KV-cache impl pre-reg scope-deferred per
  MATH §Theorem 2(b) to `exp_rdt_loop_kv_cache`.
- Elapsed: 6644 s ≈ 110.7 min. Verdict-consistency pre-flight passed
  (no antipatterns triggered; MATH.md unchanged since pre-reg; PROVISIONAL ≠
  smoke-as-full because `is_smoke=false` + under-powered target KCs is the
  F#673 path).
- PAPER.md + 4 evidence rows in DB; KCs updated (1762=pass, others=inconclusive).
- Released prior `exp_followup_cayley_riemannian_adam` claim (P3, below drain
  threshold).

## Queue state (post-iteration)
- `experiment list --status open` with P≤2: **none** (drain threshold met).
- `experiment list --status active`: **empty** (bench now `provisional`).
- Remaining open: P3+ (out of scope for current drain loop).

## Follow-ups flagged in PAPER.md (not yet filed in DB)
- `exp_rdt_loop_kv_cache` — implement & verify K-KVCACHE; unlocks full-n eval.
- `exp_rdt_loop_mmlu_eval` — K1741 with thinking preserved, 57 subjects.
- `exp_rdt_loop_gsm8k_fulln` — re-eval K1740/K1742 at pre-reg n via KV-cache.

## Hand-off
2026-04-19 drain check re-confirmed after `learning.complete`:
`experiment list --status open` P≤2 = 0; `--status active` = empty.
Printed termination signal `RESEARCH_BACKLOG_DRAINED`. No event emitted
(researcher.md step 2 terminal branch; do not emit further events).
