BitNet-SOLE Research Loop — ternary base + ternary adapters, all local.

ARCHITECTURE: BitNet-SOLE — BitNet-2B-4T ternary base + ternary LoRA adapters (QAT with STE) + 1/N composition.
STATE: Read .ralph/current_direction.md for active phase. Read HYPOTHESES.yml (bitnet section) for full roadmap.
CONTEXT: VISION.md, FINDINGS.md, references/BITNET_SOLE_RESEARCH.md, plans/concurrent-swimming-tarjan.md.

SCALE CONSTRAINT: micro only. Everything runs LOCALLY on Apple Silicon via MLX. No RunPod. No GPU costs. $0.

HOW TO PICK WORK:
1. Read HYPOTHESES.yml — find `open` nodes tagged `bitnet` with `depends_on` all satisfied
2. Priority order: P1 (foundation/critical-path) first, then P2 (scaling), then P3 (exploratory)
3. Follow the dependency chain: convergence → multiseed → scale → paper experiments
4. Set chosen node to `active`, run experiment, write PAPER.md

NOTEBOOKLM-FIRST RULE (MANDATORY):
Before implementing ANY experiment, the researcher MUST consult prior art:
- Check references/BITNET_SOLE_RESEARCH.md for related work (MoTE, LoTA-QAF, MoLoRA)
- Check references/REFERENCES.yml for matching node IDs
- Search web for "[experiment topic] ternary model" or "[experiment topic] BitNet"
- If a standard tool/framework exists, USE IT instead of reimplementing

KEY RESOURCES:
- microsoft/bitnet-b1.58-2B-4T on HuggingFace (the base model)
- micro/models/bitnet_2b_real_composition/run_experiment.py (proven MLX training pipeline)
- micro/models/bitnet_ternary_adapter_composition/ (STE training for ternary adapters)
- tools/serverless_eval.py (MATH-500 + MMLU grading logic to reuse)
- tools/orthogonality.py (adapter cosine diagnostic)
- micro/models/grassmannian_expert_init/ (AP packing, pure numpy)

DATA SOURCES (HuggingFace, $0):
- Medical: lavita/medical-instruction-tuning-datasets
- Code: codeparrot/github-code-clean
- Math: rasbt/math_distill
- Legal: nguha/legalbench
- Finance: sujet-ai/Sujet-Finance-Instruct-177k
- General: HuggingFaceH4/ultrachat_200k

PRIOR FINDINGS THAT TRANSFER (cite these):
- 1/N scaling resolves composition catastrophe (macro proven)
- Ternary adapters compose 4.4% better than FP16 (micro proven, 3 seeds)
- LOO PPL ranks adapter contribution (macro supported)
- PPL-probe weighting r=0.990 oracle correlation (micro proven)
- Safety bound alpha=0.022 (micro proven, Pre-RMSNorm architecture)
- Adaptive rank r_99/r_95 heuristic (micro proven)

RULES:
- Each experiment <2hrs. If stuck, wrap partial results and move on.
- BASELINE-FIRST: check references/ and web BEFORE implementing.
- NO new hypotheses unless all P1 nodes are complete.
- Use `uv run` for Python. Use MLX for training/inference.
- Ternary adapters (QAT+STE) are the DEFAULT, not FP16 LoRA.
