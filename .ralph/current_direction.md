# Current Direction (updated 2026-04-28)

## What happened
Three core P1/P2 experiments completed in rapid succession:
1. **Composition KILLED** (#825): Uniform Σ(A_i@B_i)/N degrades math -10pp, code -12pp.
2. **Hard routing KILLED** (#826): TF-IDF router 100% accurate but single-adapter routing (68.9%) LOSES to uniform composition (72.2%). Adapters are complementary.
3. **Interference confirmed** (#827): 3x3 matrix shows real interference (python→math -14pp, medical→code -12pp) but also surprising positive transfers (python→MedQA +50pp).

## Key insight
Adapters are both complementary (removing any one hurts) AND interfering (each one degrades specific other domains). This means **soft routing with learned per-domain weights** is the next hypothesis — neither uniform composition nor hard routing is optimal.

## Available assets
- **3 trained adapters** (math, code, medical): LoRA r=6 on q_proj, ~5MB each
- **Baselines**: GSM8K 50→72% (+22pp), HumanEval 22→70% (+48pp), MedQA 6→68% (+62pp)
- **Base model**: `mlx-community/gemma-4-e4b-it-4bit`, mlx-lm 0.31.2

## Priority for next experiments
1. **Soft routing / weighted composition** — learn per-domain weights α_i for Σ(α_i · A_i @ B_i)
2. **v_proj+o_proj adapters** (F#627 targets) — may reduce interference from q_proj adapters
3. Drain remaining P3+ backlog

## Do NOT
- Write preemptive kills, precondition probes, or documentation-only experiments
- Create antipattern memories or taxonomies
- Claim experiments that need 5+ domains until finance/legal are trained
