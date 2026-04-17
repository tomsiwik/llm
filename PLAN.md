# Plan

> Central working document. Framework principles (stable) + current research focus (iterable). Update here, not in scattered vision files.

---

## Part 1 — Framework (stable)

### What this repo is
A structured experiment framework. Hypotheses are claimed, run, reviewed, and recorded in a queryable database. The framework itself is domain-agnostic (see `README.md`); the current research instance happens to be Pierre (Part 2).

### Experiment lifecycle
```
claim → MATH.md (theorem + predictions + KC) → run_experiment.py
     → results.json → PAPER.md (prediction vs measurement)
     → REVIEW-adversarial.md → experiment complete → finding-add
```

Three hats automate this via `ralph.yml`:

| Hat | Role | Produces |
|---|---|---|
| 🔬 Researcher | design + run experiments | `MATH.md`, `run_experiment.py`, `PAPER.md` |
| 🔴 Reviewer | adversarial check | `REVIEW-adversarial.md`, verdict routing |
| 🧠 Analyst | synthesis | `LEARNINGS.md`, new `type: fix` memories |

State lives on disk (`.ralph/`, `micro/models/<exp>/`). Memories auto-inject at every hat activation (2000-token budget); new antipatterns become memories and propagate.

### Proof-first research (constructive mathematics)
Every experiment requires a formal proof **before** code:

1. **Identify the failure mode** — what specific degenerate behavior is prevented?
2. **Cite prior math** — JL-lemma, Welch bound, contractions, etc. No analogies.
3. **Derive a guarantee** — theorem/lemma that makes the failure impossible or bounds it.
4. **Predict specific numbers** — the proof makes quantitative predictions.
5. **Pre-register kill criteria** — thresholds come from the proof, not from arbitrary choice.

Three experiment types:
- **Verification** — proof complete, experiment confirms predictions.
- **Guided exploration** — proven framework, unknown parameter within it.
- **Frontier extension** — proven result being extended into new territory; mark the gap.

### Behavioral outcomes over metrics
PPL, cosine, accuracy are proxies. PPL does not predict task quality in this project (measured r≈0.08). The real questions: does the system produce useful output? does it advance the vision? A metric improving without behavioral progress is not a finding.

### Kill-criteria discipline
Pre-register K1/K2/K3 in `MATH.md` before the first run. If v1 data falsifies a KC, the status is `killed` — not "criterion reformulated". If the KC needs to change, design a v2 experiment with the new KC; don't edit the old one in place.

### Verdict consistency
Before `experiment complete --status supported`, the following must all hold:

1. `results.json["verdict"]` is not `"KILLED"`.
2. `results.json["all_pass"]` is `True` (if present).
3. `PAPER.md` verdict line does not contain `PROVISIONAL`, `PARTIALLY SUPPORTED`, `NOT SUPPORTED`, `INCONCLUSIVE`, `DEGENERATE`.
4. `is_smoke: true` runs complete as `provisional`, never `supported`/`killed`.
5. No KC was modified between MATH.md (git history) and now.
6. No auto-injected `type: fix` antipattern applies to the code.

### Antipattern catalog
Kept as `type: fix` memories in `.ralph/agent/memories.md` (auto-injected). Current set: composition math bugs, tautological routing, unsafe adapter scales, KC-swap-after-failure, verdict-DB mismatch, smoke-as-full, tautological KCs, thinking-mode truncation, wrong-model proxy, synthetic padding, `shutil.copy` as new adapter, hardcoded `"pass": True`, file-existence cache, copy-paste scaffolding, dispatch-kill mislabel. Analyst appends new ones as the loop discovers them.

### SIGREG reasoning chain
Apply to every hypothesis:
- Are you treating symptoms or the disease?
- What structure makes the failure geometrically impossible?
- Derive from existing math, not from analogy.
- Each eliminated hyperparameter is one understood degree of freedom.

Anchors: LeJEPA (`arxiv:2511.08544`), LeWorldModel (`arxiv:2603.19312`).

### Forbidden experiment classes
- Information-theory analogies without LLM evidence.
- Data-structure routing analogies (skip-lists, hash rings, cuckoo, bloom) unless paper-grounded for LLM/LoRA.
- Mechanisms with no prior paper for LLM/LoRA use.

---

## Part 2 — Current focus: Pierre

> Iterable section. Update as research progresses.

### One-line
A coding agent where every conversation trains a composable domain expert (adapter), and shared adapters make the base smarter over time.

### Platform
- **Target hardware**: Apple M5 Pro 48GB. **MLX only — no CUDA, no RunPod, no torch on GPU.** The machine is unified-memory Metal-optimized; MLX is the native path and produces dramatically better code and runtime behaviour on this hardware.
- **Base model**: `mlx-community/gemma-4-e4b-it-4bit` (dev) / `mlx-community/gemma-4-26b-a4b-it-4bit` (prod).
- **Adapter approach**: PoLAR r=6 on `v_proj+o_proj`; Grassmannian A-matrices; exclusive routing.

### Required skills (MUST invoke before writing any MLX code)
The hat loop has specialised skills that enforce idiomatic MLX and catch common mistakes. Invoke them **before coding**, not after.

| Skill | When to use |
|---|---|
| `/mlx-dev` | Any MLX array/nn/training/inference code. Enforces `mx.eval` discipline, lazy evaluation, proper module patterns, memory cleanup. |
| `/fast-mlx` | Performance-sensitive paths (training loops, inference, compile). Enforces `mx.compile`, fast ops, type promotion, bandwidth-aware kernels. |

Skipping these skills is the single biggest cause of broken MLX code in past experiments (wrong `nn.value_and_grad` mutation patterns, missing `mx.eval`, wrong BCE-with-logits handling — all appear in the audit findings). No exception: even quick smoke tests must go through `/mlx-dev`.

### Code conventions (MLX-specific)
- Use the **phased execution pattern** for memory safety: each compute phase in its own function, explicit cleanup between phases (see `/mlx-dev` skill for reference).
- `mx.clear_cache()` after loading large weights; `del` + `gc.collect()` between phases.
- No PyTorch/CUDA fallbacks. If a library requires CUDA, it's not usable here — find an MLX-native alternative or implement the primitive.
- Cite the `mlx-lm` version used in `MATH.md` — API changes between 0.21 and 0.31 have broken prior experiments.

### Working hypothesis
- Frozen base + lightweight adapters can replicate or exceed monolithic fine-tuning when adapters are (a) domain-specialized, (b) structurally orthogonal, (c) routed cleanly.
- Composition must happen in continuous space; ternary/quantized composition requires explicit handling (BitNet foundations).
- Thinking mode (Gemma 4 `<|channel>thought...`) must be preserved during training and eval, or reasoning degrades.

### Deep reference (outside repo)
The vision/architecture is maintained **outside this repo** (chat + parent dir) to keep the framework clean:
- `../VISION_P1.md` — current full architecture
- `../ARCHITECTURE_P1.md` — mathematical reference
- `../PIERRE.md` — product framing

Do **not** duplicate their content here. Summarize only what's needed for day-to-day experiment work.

### Current phase (as of 2026-04-17)
Repo-wide audit completed (62 batches, 615 experiments). Confirmed systemic issues: composition bug shipped in v1 code, tautological routing in Pierre v3–v6, LORA_SCALE=20 inflating claims, thinking-mode truncation giving false domain gains. Recovery work: re-validate supported findings, rerun affected composition experiments with correct `Σ B_i @ A_i` math.

DB state: 720 experiments (77 new follow-ups/Gemma 4 recreates imported), 91 status updates applied (24 resurrects + 67 reruns → open with audit-rerun tags).

### Active workstreams
- **P11** — reasoning training recipe (s1K, LIMO, GRPO, ThinkPO, Plan-and-Solve).
- **Routing** — TF-IDF+embedding logistic; N=25 at ~89% top-1 accuracy.
- **Thinking preservation** — MCQ adapters must train with `enable_thinking=True`.

### Near-term goals
- Match or beat Gemma-4 MMLU-Pro baseline (62.1% with thinking) using an adapter that doesn't suppress thinking.
- Verify Grassmannian orthogonality claim on real Gemma 4 runs (not Qwen proxy).
- Close composition-bug recovery: identify every headline number derived through the buggy path; rerun or flag.

### Pierre code progression

**Current stable:** `pierre/pierre.py` (265 loc) — runtime composition pipeline. `pierre/bench.py` — benchmarks. `pierre/math/` — predict + theoretical analysis.

**Version protocol** — each new Pierre version is two artifacts:
1. A snapshot under `pierre/archive/vN/` (frozen code).
2. A validation experiment `exp_pierre_vN_*` in the experiment DB (MATH.md + run_experiment.py + PAPER.md).

Promote a version into `pierre/pierre.py` only after its validation experiment reaches `status=supported` AND passes the verdict-consistency checklist (PLAN.md §1).

**Changelog:**

| Version | Hypothesis | Verdict | Experiment | Notes |
|---|---|---|---|---|
| v1 | original pre-merge composition | supported (retroactively flagged) | — | **composition bug**: code summed `lora_A`/`lora_B` safetensors independently → `(ΣB)(ΣA)` cross-product. Fix required in v8. |
| v3 | SFT adapters + BitLinear side-path | supported | `exp_pierre_v3_sft_n5`, `exp_pierre_v3_n24_scaling` | Used tautological routing `route(val[d][0])`. Headline "0% degradation" is an artifact. |
| v4 | ternary premerge (merge LoRA into BitLinear) | killed | `exp_pierre_v4_ternary_premerge` | Ternary has no room for merged deltas (BitNet foundations). |
| v5 | fully ternary LoRA (Grassmannian A + STE B) | supported | `exp_pierre_v5_ternary_lora` | Same tautological routing as v3. |
| v5.1 | LoTA-QAF lossless ternary merge | killed | `exp_pierre_v51_lota_merge` | |
| v5.2 | Bankai-inspired ternary row flips (greedy search) | killed | `exp_pierre_v52_bankai_flip` | `revert_row_flip` not actually reversible after clipping; caught pre-run. |
| v5.3 | lazy bf16 LoRA side-path | killed | `exp_pierre_v53_lazy_sidepath` | |
| v5.4 | `mx.quantized_matmul` 2-bit lazy side-path | killed | `exp_pierre_v54_quantized_matmul` | |
| v6 | precomputed concatenated deltas (attention-factored) | supported | `exp_pierre_v6_precomputed_concat` | Same tautological routing. |
| v6.2 | hybrid precomputed attention + factored MLP | killed | `exp_pierre_v62_hybrid` | |
| v7 | keyframe adapter (deterministic verifier) | killed | `exp_pierre_v7_keyframe_poc` | `phase_composition` accepts `verifier` arg but never uses it (audit finding). |
| v7.1 | keyframe with last-token hidden state | killed | `exp_pierre_v71_keyframe_lasttoken` | Same unused-verifier bug as v7. |

**Next version plan (v8 — working draft):**

Goals driven by audit recovery:
1. **Fix composition math** — `Σ B_i @ A_i` correctly; drop any code path that sums A and B tensors independently (`mem-antipattern-001`).
2. **Per-sample routing** — replace `route(val[d][0])` with per-sample routing; headline PPL must not equal single-adapter PPL by construction (`mem-antipattern-002`).
3. **Drop LORA_SCALE=20** — default to safe scale (≤ 8); if a claim requires higher scale, it's a scale-specific claim, not a general property (`mem-antipattern-003`).
4. **Thinking preservation** — training and eval both use `enable_thinking=True`; strip regex matches Gemma 4's native `<|channel>thought` format (`mem-antipattern-008`).

Validation protocol:
- Snapshot current `pierre/pierre.py` → `pierre/archive/v7/` if not already there (v7 exists, confirm).
- Write `exp_pierre_v8_*` experiment with MATH.md pre-registering KCs: (i) `composed_ppl != single_ppl` across > 1 sample; (ii) per-sample router accuracy ≥ 85% at N=5; (iii) MMLU-Pro with thinking ≥ 62.1%; (iv) no antipattern memory triggers in review.
- Do not promote v8 → `pierre/pierre.py` until `exp_pierre_v8_*` is `supported` AND the audit-flagged v3/v5/v6 numbers have been reproduced cleanly or retracted.

**Where state lives (reference):**
- Per-experiment: experiment DB (Turso) via `experiment` CLI. This is authoritative.
- Short-term ralph loop: `.ralph/current_direction.md`.
- Roadmap + version plan: this file (PLAN.md Part 2).
- Antipatterns: `.ralph/agent/memories.md` (`type: fix`), auto-injected.

---

## Iteration discipline

- **This document is the central piece.** When research direction shifts, edit Part 2 here — don't fork into a new VISION.md.
- Part 1 changes only when a framework principle has been falsified or extended (rare). Treat it as settled unless a finding forces an update.
- Part 2 changes freely. Keep it short; push deep details to parent docs or `micro/models/<exp>/PAPER.md`.
- New antipatterns → memory entries (`.ralph/agent/memories.md`, `type: fix`), not here.
