BitNet-SOLE Research Loop — Wave 3: Foundation + Base-Free + Serving + Evolve.

ARCHITECTURE: Ternary base (BitNet-2B-4T) + ternary LoRA adapters (QAT+STE) + Grassmannian skeleton (frozen A, 17x decorrelation filter) + runtime LoRA serving (no merge).
STATE: Read .ralph/current_direction.md for active work. Read HYPOTHESES.yml Wave 3 section.
CONTEXT: VISION.md (rewritten 2026-03-22), FINDINGS.md, references/BITNET_SOLE_RESEARCH.md.

SCALE CONSTRAINT: micro only. LOCAL Apple Silicon via MLX. $0.

ORPHAN CHECK (do FIRST):
- Read .ralph/current_direction.md to find the last experiment worked on.
- Check if its REVIEW-adversarial.md and LEARNINGS.md exist.
- If either is missing, resolve the orphan before picking new work.

4 PARALLEL TRACKS (work on the highest-priority unblocked node):

Track 1 — Foundation Fixes (P1):
  exp_bitnet_effective_delta_cosine (P1, OPEN) → measure vec(B@A) cosine, fix orthogonality.py
  exp_bitnet_kr_test_evaluation (P1, OPEN) → knowledge retention metric replaces PPL
  exp_bitnet_lori_sparse_b (P2, OPEN) → 90% B-sparsity for interference reduction

Track 2 — Base-Free Scaffold (P1):
  exp_bitnet_scaffold_fresh_adapters (P1, OPEN) → train ON random scaffold (prior test was wrong)
  exp_bitnet_galore_scaffold (P1, OPEN, no deps) → GaLore-grown base from scratch
  exp_bitnet_meta_scaffold (P3, OPEN) → MAML meta-learned scaffold (novel, no prior work)

Track 3 — Production Serving (P2):
  exp_bitnet_llamacpp_serving (P2, OPEN) → llama.cpp --lora multi-adapter CPU
  exp_bitnet_per_token_routing (P2, OPEN) → MoLoRA-style per-token routing

Track 4 — Evolve Redesign (P2):
  exp_bitnet_retrain_evolve (P2, OPEN) → retrain-from-scratch + KR-Test quality gate

PRIORITY ORDER: P1 nodes first across all tracks. Within same priority, prefer Track 1/2 over 3/4.
The two P1 nodes with NO dependencies are: exp_bitnet_galore_scaffold and exp_bitnet_effective_delta_cosine.

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
- BASELINE-FIRST: NotebookLM + references/ BEFORE implementing.
- Every result gets: adversarial review → analyst LEARNINGS.md → THEN next experiment.
- Check for orphan experiments (missing REVIEW/LEARNINGS) before starting new work.
- Use `uv run` for Python. Use MLX for training/inference.
