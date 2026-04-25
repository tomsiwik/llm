# REVIEW-adversarial — exp_g4_adapter_class_composition_full_impl

**Verdict: PROVISIONAL** (Phase A executable slice; structural-KC PASS, target-KC `untested`)

## Adversarial checklist (25 items)

| # | Item | Status | Note |
|---|------|--------|------|
| a | results.json verdict vs DB status | PASS | both `PROVISIONAL` |
| b | all_pass vs claim | PASS | `null`; no all-pass asserted |
| c | PAPER.md verdict line | PASS | `PROVISIONAL` matches |
| d | is_smoke vs claim | PASS | `false`; honest Phase A slice (not a full-run mislabel) |
| e | KC mutation post-claim | PASS | K1-K4 verbatim from parent (F#686), no relax |
| f | Tautology sniff | PASS | KCs `untested`; A1-A3 are honest inspection (load, count, symbol-grep) |
| g | K-ID code↔math | PASS | Phase A code measures only A1-A3; K1-K4 untouched per scope |
| h | Bad composition idiom | PASS | no composition code (Phase A is inspection) |
| i | LORA_SCALE ≥ 12 | PASS | no LoRA training in this iter; default `6.0` documented in MATH.md §0 |
| j | Single-sample routing | PASS | no routing |
| k | shutil.copy aliasing | PASS | none |
| l | Hardcoded `{"pass": True}` | PASS | A1-A3 statuses populated from real inspection (`mlx_lm.load`, `dir(lora_mod)`) |
| m | Target model ↔ loaded model | PASS | both `mlx-community/gemma-4-e4b-it-4bit` |
| m2 | Skill invocation | PASS | MATH.md §0 cites `/mlx-dev`, `/fast-mlx`; code uses `mlx_lm.load`, `mx.clear_cache()` (idiomatic for inspection scope) |
| n | base 0% + thinking_chars=0 | N/A | no eval |
| o | Headline n<15 | N/A | no eval |
| p | Synthetic padding | N/A | no eval |
| q | Cited-baseline drift | N/A | no eval |
| r | Prediction-vs-measurement table | PASS | PAPER.md §"Phase A executable slice" + §K1-K4 prediction-vs-measurement |
| s | Math errors / unsupported claims | PASS | A3 FAIL is reported as symbol-level only, not over-claimed at CLI level |
| t | Target-gated kill (F#666) | N/A | PROVISIONAL not KILL |
| u | Scope-changing fix | PASS | Phase A executable slice + Phase B-E deferred is *honest* design + executable inspect, not silent scope-reduction; F1-F5 forbid list explicit |
| PROVISIONAL gate | Phase A executable slice precedent | PASS | F#772 (jepa) + F#799 (memento) precedent; identical artifact pattern |
| F#666 target-gating | matrix preserved | PASS | matrix unchanged from parent §3 |
| F#669 cascade | parent-gating check | N/A | Phase A is pure inspection, no schema-broken deps |

**Result: 25/25 PASS or N/A.** Block-fix count: 0.

## Phase A readout (load-bearing for future Phase B)
- A1 base loads: PASS — `mlx-community/gemma-4-e4b-it-4bit`, mlx-lm 0.31.2, 5.49s wall.
- A2 `v_proj`+`o_proj`: PASS — **42 layers × both = 84 LoRA targets per adapter**; F#627 confirmed at Gemma 4 E4B 4-bit scale.
- A3 DoRA available: **FAIL** — 0 dora-related symbols in `mlx_lm.tuner.lora` at v0.31.2. Parent §0 assumption (`--fine-tune-type dora`) is *symbol-level unverified*; CLI-level still pending. Implication: B1 scope may need 2 custom modules (DoRA + MoLoRA), not 1.

## Why PROVISIONAL not SUPPORTED
- `is_phase_a_executable_slice=True` caps verdict at PROVISIONAL by precedent.
- K1-K4 all `untested` — Phase B-E deferred per single-iter cap (8-15h estimated).
- A3 FAIL is honest signal, not a pre-reg KC failure.

## Why PROVISIONAL not KILLED
- A1+A2 PASS (substrate for future Phase B exists at the topology level).
- A3 FAIL is informational (DoRA assumption needs CLI-level re-verification), not a structural KC failure.
- Parent F#686 is design-only PROVISIONAL; this Phase A slice extends it constructively.
- F#666 target-gating preserved (no proxy-PASS asserted as target-PASS, no kill on proxy without target-FAIL).

## Drain-window milestone
This is the LAST P≤2 drain pick. Verified post-iter: P≤2 open=0, active=0 ⇒ `RESEARCH_BACKLOG_DRAINED` met.

## No `_impl` companion filed
Same precedent as `exp_memento_gemma4_replication_impl` (F#799), `exp_jepa_adapter_residual_stream_impl` (F#772): the `_impl` *is* the impl follow-up of the parent design-only PROVISIONAL. Phase B-E execution at P=3 reuses this same directory; no new task entry needed.

## Assumptions
- A3 symbol-level absence does not preclude CLI-level support; future Phase B must verify with `mlx_lm.lora --fine-tune-type dora` before deciding LoRA-vs-DoRA path.
- 42-layer × {v_proj, o_proj} topology is stable across mlx-lm 0.31.x (verified 0.31.2 only).
