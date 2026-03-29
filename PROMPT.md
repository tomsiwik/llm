BitNet-SOLE Research Loop — Continuous experimentation on Apple Silicon.

PROOF-FIRST RESEARCH (Constructive Mathematics):
  Before writing code, run the SIGReg Reasoning Chain:
  a) Am I treating SYMPTOMS or the DISEASE? If adding 3rd+ fix → STOP, find root cause
  b) Reframe: not "prevent X" but "what optimal structure makes X impossible?"
  c) Derive from EXISTING math (JL-lemma, Welch, concentration inequalities, etc.)
  d) Write Theorem/Proof/QED in MATH.md with quantitative + behavioral predictions
  e) Run experiment to VERIFY the proof — kill criteria derived from proof
  f) MATH.md must end with completed Self-Test (6 questions)
  Three types: verification (proof complete), guided exploration (unknown params
  in proven framework), frontier extension (extending proven math to new territory).
  Ref: LeJEPA (2511.08544), LeWorldModel (2603.19312).

ARCHITECTURE: Ternary base (BitNet-2B-4T or OUR OWN ternary base) + composable perturbation operators (not limited to LoRA) + structural guarantees against interference + performant inference on Apple Silicon.
STATE: Read .ralph/current_direction.md for active work.
CONTEXT: VISION.md, `experiment finding-list`, `experiment query`.

## TARGET PLATFORM (HARD CONSTRAINT)
- Apple M5 Pro, 48GB unified memory, MLX 0.31.1
- This IS the deployment target. Not a stepping stone. Not "micro scale."
- All experiments MUST run on Apple Silicon via MLX. No CUDA. No RunPod.
- The "macro scale" = edge of what M5 Pro 48GB can do (~40GB usable).
- A final composed model must serve interactively within this envelope.
- Pre-merge composition is FREE (0.80% overhead). Always pre-merge on MLX.

## STRATEGIC PRIORITIES (March 2026)

### P0: DEPLOYMENT TRACK — Ship a working system on BitNet-2B-4T (HIGHEST PRIORITY)
Use Microsoft's BitNet-2B-4T as-is. Train real-data adapters. Route. Compose. Generate.
Prove the architecture produces BETTER TEXT, not just lower PPL.

**Critical path (in order):**
1. exp_generation_quality_test — does routed composition produce better text? THE existential test.
2. exp_task_accuracy_real_benchmarks — MMLU/GSM8K/HumanEval with composition.
3. exp_real_data_25_domain_adapters — scale to 25 real instruction-tuned adapters.
4. exp_e2e_demo_pipeline_mlx — full pipeline: query → route → compose → generate.
5. exp_dynamic_adapter_addition — hot-add new adapters without retraining.

**What we already have for this track:**
- BitNet-2B-4T base (1.7GB, free, available)
- 5 real-data instruction-tuned adapters (medical/code/math/legal/finance, -26.3% PPL)
- Gumbel-sigmoid routing (44% better than softmax, 0.58% overhead)
- Tiny routing heads (100% accuracy on 5 domains, 2.32% overhead)
- Entropy gating pre-filter (63% tokens skip at 1.13% cost)
- N-gram cache (20.8% free PPL reduction, 14.5MB)
- Pre-merge serving (0% overhead on MLX)
- Fully ternary adapters (15.8x compression, pure addition)

### P1 (PARALLEL): Train Our Own Ternary Base (Research Track)
This is a research goal, NOT a blocker for the deployment track.
- Training ternary models from scratch on MLX using STE
- Falcon-Edge's onebitllms toolkit (tiiuae/onebitllms) — Triton kernels for STE
- MatMul-free LM architecture (ridgerchu/matmulfreellm) — ternary + no matmul
- GaLore+STE integration (our GaLore experiment showed 2-3x ternary degradation
  without STE-in-loop; STE-aware GaLore is the fix)
- Sparse-BitNet (arxiv 2603.05168) — 42% natural sparsity in ternary weights
- Tequila (arxiv 2509.23800) — fixes BitNet deadzone trapping via Minima Reactivation
- Sherry (Tencent/AngelSlim) — 1.25-bit with 3:4 sparsity, SIMD packing
- GOAL: A ternary base we control, trained on M5 Pro, supporting composition

### P0: Base-Free via Tiny Per-Adapter Routing Heads
Random scaffold is dead (PPL 319M). But explore a different angle:
- Each adapter carries its own tiny "routing head" (~5K params)
- Entropy-adaptive gating: skip experts when base model is confident
- Parameter-golf insight: n-gram cache + entropy mixing gives 15%+ gain
- Sakana AI Text-to-LoRA (SakanaAI/text-to-lora) — hypernetwork generates
  task-specific adapters from text description in single forward pass
- PiSSA initialization (arxiv 2404.02948) — principal SVD for better LoRA init
- ZipIt! (gstoica27/ZipIt) — cross-init model merging via feature permutation
- L2R (Learning to Route) — Gumbel-sigmoid non-competing multi-adapter routing
- GOAL: composition that doesn't need a fixed base, or minimizes base dependency

### P1: Test-Time Training for Expert Selection
From parameter-golf #1 entry (1.1194 BPB):
- TTT: adapt weights per-document at inference time
- Protocol: score input → compute entropy → TTT-adapt relevant experts → generate
- Sidesteps learned router entirely — model self-selects knowledge
- TTT Done Right (arxiv 2505.23884) — reference implementation
- GOAL: runtime expert selection without router overhead

### P1: Mechanism Story (What ACTUALLY Makes Composition Work?)
- exp_bitnet_effective_delta_cosine: measure vec(B@A) cosine, not just A-cosine
- If effective-delta cos is low: Grassmannian IS the mechanism (load-bearing)
- If effective-delta cos is high but composition works: constructive transfer
- OSRM (arxiv 2505.22934) showed weight-space ≠ data-space orthogonality
- GOAL: settle the mechanism question definitively

### P2: Production Serving on Apple Silicon
- Pre-merge is the answer (0% overhead proven)
- Per-token routing via MoLoRA (arxiv 2603.15965): Qwen3-1.7B+4 adapters > 8B
- X-LoRA (EricLBuehler/xlora) — dense mixing of LoRA experts, in HF PEFT
- EdgeLoRA — multi-tenant LoRA serving on edge, intelligent caching
- CLONE (arxiv 2506.02847) — MoE router for dynamic LoRA selection at edge
- GOAL: interactive serving on M5 Pro with dynamic expert selection

## KEY NEW REFERENCES (from parameter-golf + web research)

- Parameter Golf: https://github.com/openai/parameter-golf
  - TTT is biggest lever (#1: 1.1194 BPB via per-doc adaptation)
  - MoE fails below 500M params (Apple scaling laws, ICML 2025)
  - Ternary quant competitive at rank #10 (1.1570 BPB, 73.7M params)
  - N-gram + entropy mixing: 0.9674 BPB (15%+ over neural alone)
  - XSA (Exclusive Self-Attention): zero-param attention improvement
  - Partial RoPE (25% dims): position-free dims as routing features
- Falcon-Edge: tiiuae/onebitllms — open ternary training toolkit
- MatMul-free LM: ridgerchu/matmulfreellm — ternary + no matmul, up to 2.7B
- MLX-BitNet: exo-explore/mlx-bitnet — first ternary impl for Apple Silicon
- Text-to-LoRA: SakanaAI/text-to-lora — hypernetwork generates LoRA from text
- X-LoRA: EricLBuehler/xlora — mixture of LoRA experts in HF PEFT
- MoLoRA: arxiv 2603.15965 — per-token routing, 1.7B beats 8B
- Sparse-BitNet: arxiv 2603.05168 — natural 42% sparsity in ternary weights
- Cross-LoRA: arxiv 2508.05232 — data-free LoRA transfer across base models
- PiSSA: arxiv 2404.02948 — SVD-init LoRA, NeurIPS 2024 spotlight
- LD-MoLE: arxiv 2509.25684 — learnable dynamic routing for MoLoRA experts
- M5 Neural Accelerators: Apple MLX research — up to 4x speedup over M4

## EXPERIMENT SELECTION (do this EVERY iteration):
1. Run: `experiment list --status open,active` to see available work.
2. Run: `experiment list --blocking` to find critical-path experiments.
3. Pick the highest-priority unblocked experiment. Use `experiment get <id>` for full details.
4. Run: `experiment update <id> --status active` before starting work.
5. If NO open/active experiments exist, generate new hypotheses from this file and `experiment finding-list` + `experiment query`.
6. Only output RESEARCH_BACKLOG_DRAINED when no actionable experiments remain.

AFTER COMPLETING AN EXPERIMENT:
- `experiment update <id> --status supported` (or `proven` or `killed`)
- `experiment evidence <id> --claim "K1 PASS: metric=value" --source "results.json" --verdict pass`

ORPHAN CHECK (do FIRST):
- Read .ralph/current_direction.md to find the last experiment worked on.
- Check if its REVIEW-adversarial.md and LEARNINGS.md exist.
- If either is missing, resolve the orphan before picking new work.

NOTEBOOKLM-FIRST RULE (MANDATORY):
Before implementing ANY experiment, consult NotebookLM (/notebooklm):
- Query: "What methods exist for [this problem]? What pitfalls? What implementations?"
- Check references/BITNET_SOLE_RESEARCH.md for related work
- If a standard tool/framework exists, USE IT instead of reimplementing
- Web search is FALLBACK only for topics not in the notebook

KEY RESOURCES:
- micro/models/bitnet_2b_real_composition/run_experiment.py (MLX training pipeline)
- micro/models/bitnet_ternary_adapter_composition/ (STE training)
- tools/orthogonality.py (needs --effective-delta mode added)
- tools/serverless_eval.py (grading logic to reuse)
- micro/models/grassmannian_expert_init/ (AP packing, pure numpy)

DATA SOURCES (HuggingFace, $0):
- Medical: lavita/medical-instruction-tuning-datasets
- Code: codeparrot/github-code-clean
- Math: rasbt/math_distill
- Legal: nguha/legalbench
- Finance: sujet-ai/Sujet-Finance-Instruct-177k
- General: HuggingFaceH4/ultrachat_200k

RULES:
- KEEP GOING. After each cycle, pick the next experiment. Never stop early.
- Each experiment <2hrs. If stuck, wrap partial results and move on.
- ALL experiments on MLX/Apple Silicon. No CUDA. No RunPod.
- BASELINE-FIRST: NotebookLM + references/ BEFORE implementing.
- Every result gets: adversarial review → analyst LEARNINGS.md → THEN next experiment.
- Check for orphan experiments (missing REVIEW/LEARNINGS) before starting new work.
- Use `uv run` for Python. Use MLX for training/inference.
- Use the `experiment` CLI for ALL experiment state management.
- Use `experiment claim <worker>` to pick work, `experiment complete <id> ...` to finish.
- Invoke /fast-mlx and /mlx-dev skills before writing ANY MLX code.
