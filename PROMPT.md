BitNet-SOLE Research Loop — Continuous experimentation.

ARCHITECTURE: Ternary base (BitNet-2B-4T) + ternary LoRA adapters (QAT+STE) + Grassmannian skeleton (frozen A, 17x decorrelation filter) + runtime LoRA serving (no merge).
STATE: Read .ralph/current_direction.md for active work.
CONTEXT: VISION.md, FINDINGS.md, references/BITNET_SOLE_RESEARCH.md.

SCALE CONSTRAINT: micro only. LOCAL Apple Silicon via MLX. $0.

EXPERIMENT SELECTION (do this EVERY iteration):
1. Run: `experiment list --status open,active` to see available work.
2. Run: `experiment list --blocking` to find critical-path experiments.
3. Pick the highest-priority unblocked experiment. Use `experiment get <id>` for full details.
4. Run: `experiment update <id> --status active` before starting work.
5. If NO open/active experiments exist at matching scale, generate new hypotheses from FINDINGS.md and add via `experiment add`.
6. Only output RESEARCH_BACKLOG_DRAINED when `experiment list --status open,active` returns zero rows AND you have generated new hypotheses but none are actionable.

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
- BASELINE-FIRST: NotebookLM + references/ BEFORE implementing.
- Every result gets: adversarial review → analyst LEARNINGS.md → THEN next experiment.
- Check for orphan experiments (missing REVIEW/LEARNINGS) before starting new work.
- Use `uv run` for Python. Use MLX for training/inference.
- Use the `experiment` CLI for ALL experiment state management. Do NOT edit HYPOTHESES.yml directly.
