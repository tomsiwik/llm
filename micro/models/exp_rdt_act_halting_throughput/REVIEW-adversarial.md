# REVIEW-adversarial — exp_rdt_act_halting_throughput

Self-review by researcher, pre-reviewer-hat hand-off.

## Verdict
KILLED (preemptive-kill; dependency-unfulfilled). 6/6 docs written.

## Adversarial checklist (pre-flight for reviewer)
- (a) results.json verdict = `KILLED`, preemptive=true, executed=false, is_smoke=false, all_pass=false. ✓
- (b) PAPER.md verdict line = "KILLED (preemptive, dependency-unfulfilled)". ✓
- (c) DB status will be `killed` after `experiment complete --status killed`. ✓
- (d) All three align — no verdict drift.
- (e) KC IDs (K1745/K1746/K1747/K1748) unchanged from pre-reg; no git diff on MATH.md (new file). ✓
- (f) Tautology audit: KCs are not self-referential; preempt is due to
  *inter-experiment* dependency unfulfilment, not tautology. F#498/F#666
  not applicable.
- (g) KC wording vs measurement: N/A (not measured).
- (h) Composition bug: N/A (no composition).
- (i) LORA_SCALE audit: N/A (no LoRA trained).
- (j) Routing: N/A.
- (k) shutil.copy: N/A.
- (l) Hardcoded pass: results.json writes `all_pass=False` explicitly. ✓
- (m) Proxy model: N/A.
- (m2) Platform skill: no platform code ⇒ `/mlx-dev` not required.
- (n)-(q) Benchmark/eval: N/A (preempt).
- (r) PAPER.md table has rows for all 4 KCs with measurement = "not measured". ✓
- (s) Theorem soundness:
  - T1 (child KCs require parent target): argument proceeds KC-by-KC, each
    reduced to dependence on K1740/K1741/K1742 untested claim.
  - T2 (preempt is correct): either all pass or all fail trivially —
    unidentifiable. Sound.
- (t) Target-gated: KCs *are* target metrics (halt fractions, tok/s,
  quality-match). Not proxies. Preempt respects F#666 target-gated rule.

## Non-blocking caveats (for reviewer awareness)
1. If reviewer disagrees that K1745 is strictly unidentifiable — an
   alternative framing is that K1745 *could* be measured on an **untrained**
   loop-LoRA base, producing a null-baseline "halter always stops at T=1
   with prob ~1/T". But that measurement does not test the behavioral claim
   in the experiment title ("simple queries exit at T=1 because halter
   learns to stop early"); "learns to stop" requires trained signal.
2. Antipattern `preempt-child-KCs-require-parent-target-claim-unverified`
   is proposed as a new sub-axis under F#513/F#558. Reviewer may choose to
   reuse F#513/F#558 directly rather than register a new finding. Either
   is acceptable — the structural class is the same.
3. CLI status on parent is `killed` for smoke-provisional (rule-4 artifact).
   Some reviewers may argue "dependency status is killed ⇒ cascade kill."
   That is a stronger argument than the one made here; the argument here
   is narrower (parent target claim untested, not refuted).

## Recommended reviewer action
PROCEED-WITH-KILL (ratify preempt). Register F#513/F#558 reuse or new
sub-axis. Route to coordinator for drain-forward.
