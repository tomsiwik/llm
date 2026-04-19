# exp_rdt_depth_domain_2d — REVIEW-adversarial.md (researcher self-review)

## Checklist (a)-(t)

- (a) results.json verdict = KILLED; DB status → killed; PAPER.md verdict = KILLED. MATCH.
- (b) all_pass = false. Consistent.
- (c) preemptive=true, executed=false, is_smoke=false. No smoke/full confusion.
- (d) KC results: all 4 = fail/not_measured. No claim upgraded.
- (e) KC text matches DB KC text verbatim (K1749–K1752).
- (f) Not tautology — dep-unfulfilled via inter-experiment chain, not
  measuring-something-we-baked-in (F#498/F#666).
- (g) N/A — nothing measured.
- (h) No composition math executed ⇒ no Σ-A-Σ-B bug possible.
- (i) No LORA_SCALE — no training.
- (j) No routing executed.
- (k) No shutil.copy — run_experiment.py only writes results.json.
- (l) No hardcoded pass — all KCs explicitly fail.
- (m) Target model would be Gemma 4 E4B (per parent); not loaded because preempt.
- (m2) No platform code ⇒ /mlx-dev not required. G1011 OK.
- (n)-(q) N/A (no eval).
- (r) PAPER.md prediction-vs-measurement table present with all 4 rows marked
  "not measured" + reason citing Theorem.
- (s) MATH.md Theorems 1-4 are structural dep-reductions; each cites a
  concrete prior finding (F#668, F#571, F#562, F#669) and base fact
  (LoRA B=0 at init).
- (t) KCs are target-type (task quality, saturation curve, cos on trained ΔW,
  Room Model identity). Not being proxy-killed — being dep-unfulfilled killed.

## Reviewer caveats (non-blocking)

1. **F#669 second reuse** — on next occurrence, promote from sub-axis.
   Precedents: exp_rdt_act_halting_throughput (1st), exp_rdt_depth_domain_2d (2nd).
2. **Double-parent-unfulfilled** is a variant: both parents are in
   non-trained-artifact states, not just one. Captured in MATH.md but does
   not change the verdict.
3. **K1752 is redundant** — even with trained artifacts, F#571 (Room Model
   superseded for N>1) would kill it. Could mark K1752 as "superseded" separately
   in the DB finding, but doesn't change this experiment's outcome.
4. **Unblock depends on macro compute** — `exp_rdt_loop_lora_gemma4_full` is
   the follow-up ticket that needs real GSM8K+MATH training. This is outside
   the current P≤2 drain budget.

## Verdict

KILLED (preemptive). Artifacts complete (6/6). No scientific kill — scope-
bounded by dependency chain. Drain-forward.
