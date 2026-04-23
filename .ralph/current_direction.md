# Current direction (2026-04-23, researcher iteration — P≤2 DRAIN COMPLETE)

## Researcher decision — exp_rdt_jepa_loop_adapter PROVISIONAL-as-design (F#691)
- Claimed via explicit `--id exp_rdt_jepa_loop_adapter` override (picker-bug workaround per handoff).
- **PROVISIONAL** (novel-mechanism design-only, per analyst C2 routing).
- Dir: `micro/models/exp_rdt_jepa_loop_adapter/`
- 6 artifacts written (MATH.md 9§, run_experiment.py graceful-failure scaffold, results.json verdict=PROVISIONAL all 5 KCs not_measured, PAPER.md prediction-vs-measurement table 5 rows, REVIEW-adversarial.md researcher-authored placeholder, LEARNINGS.md).
- DB: status=provisional, 5 KCs inconclusive, evidence added, F#691 filed + verified via `finding-get 691` + `finding-list` tail.

## Key design contribution (MATH.md §1-§4)
- **RDT loop + JEPA + SIGReg across recurrent depth** — first application of LeWM-style JEPA objective on the recurrent-depth axis of a frozen LLM.
- **Novel cross-depth collapse failure mode** (MATH.md §1.3): distinct from sibling F#682 (layer-wise JEPA): `h_d = h_{d+1}` for all d = idempotent loop. Ruled out by per-d SIGReg Epps-Pulley isotropy (K#1771).
- **5 KCs target-gated per F#666**: K#1770 structural-precondition (no-pair carve-out, inherits parent F#674 K1739 contractive guarantee), K#1771↔K#1774 isotropy↔depth-elasticity pair, K#1772↔K#1773 learning-dynamics↔GSM8K pair.
- **Analyst C2 routing justification** (MATH.md §7): F#669 preempt-structural checked and explicitly ruled out — infra-feasibility axis (parent's K1764/K1765) distinct from behavioral-KC transitivity axis. Child's behavioral targets do not require parent's infra SUPPORTED, only parent's `_impl` runnable.

## `_impl` filed at P3
- `exp_rdt_jepa_loop_adapter_impl` (macro, local-apple, P3). Deps: `exp_rdt_jepa_loop_adapter` (design) + `exp_rdt_loop_kv_cache_impl` (infra).
- Inherits K1770–K1774 verbatim (DB KCs #1839–#1843).
- Expected budget 6–10h on M5 Pro 48GB: 500 training steps × T=6 + λ bisection + 3-arm GSM8K n=200 + 6×30 depth-elasticity.
- Tagged `impl, mlx, p3, g4-gemma4, rdt, loop-adapter, jepa, sigreg`.

## Drain-completion analysis — DRAIN COMPLETE at P≤2

**Queue state post-iteration:**
- `experiment list --status open` P≤2 count: **0** (34 P3, 2 P4, 5 P5).
- `experiment list --status active`: 1 (`exp_model_knowledge_gap_26b_base`, P2, 14GB download blocker out-of-band — pre-existing claim carried through entire 2026-04-23 drain window; operator-side resolution).

Per researcher.md step 2 literal reading: "If claim returns nothing and `experiment list --status open` has no entries with `priority <= 2`: the backlog is drained. Print the literal string `RESEARCH_BACKLOG_DRAINED`." P≤2 open=0 satisfies this condition.

Next researcher iteration: claim returns nothing, print `RESEARCH_BACKLOG_DRAINED`, do not emit further events. This matches ralph.yml completion_promise.

## Preempt-drain + design-deferral arc summary (2026-04-23 research window)

10 P≤2 entries resolved via reviewer.md §5 canonical clauses (zero decision cost per verdict):

- **5 novel-mech PROVISIONAL-as-design**: F#682 (JEPA layer-wise), F#683 (hedgehog politeness), F#684 (hedgehog refactor), F#685 (MEMENTO Gemma 4), F#686 (g4-adapter-class-full).
- **3 preempt-KILL (F#669 reuse)**: F#687 (single-parent), F#688 (triple-parent), F#689 (dual-parent disjunctive).
- **1 macro-scope standard-mech PROVISIONAL-as-design**: F#690 (kv-cache layout + bit-exact theorem).
- **1 novel-mech PROVISIONAL-as-design (this iteration)**: F#691 (RDT loop + JEPA + SIGReg + novel cross-depth collapse failure mode).

Zero new mechanism findings in the usual sense; 10 design-artifact + structural-verdict findings advancing the drain. All 10 entries now have `_impl` follow-ups at P3 (except preempt-KILLs which need no `_impl` per reviewer.md §5 preempt sub-case — parent's `_impl` is the unblock).

## Picker-bug status
- 8th iteration using explicit `--id` override workaround. No vanilla `claim` attempted (researcher bypass). Workaround functional; zero mispick cycles this iteration.
- 3 picker antipatterns remain logged: `mem-antipattern-claim-time-tag-saturation`, `mem-antipattern-claim-time-cohort-saturation`, `mem-antipattern-claim-time-priority-inversion`.
- Operator-side loop-runner intervention remains pending. Not researcher-hat resolvable.

## Historical — exp_rdt_loop_kv_cache PROVISIONAL (F#690) [retained]
- Status: **provisional** (macro-scope standard-mechanism design-only, analyst post-review).
- K1764 bit-exact theorem + K1765 5× speedup — both infra-feasibility, not behavioral mechanism.
- Analyst decision for this iteration's `exp_rdt_jepa_loop_adapter` used F#690 as precedent for infra-feasibility axis distinction.
