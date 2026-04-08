# Adapter Solidification: Research Findings

## The Problem
Large adapters (scale=20) destroy the base model's knowledge (-60pp MMLU on Qwen3).
Small adapters (scale≤5) preserve knowledge but may lack behavioral benefit.
The adapter and base are FIGHTING each other.

## Key Papers Found

### 1. ScaleZero — Progressive Adapter Integration
- LoRA adapters start as temporary, then get "solidified" into MoE backbone
- Dynamic Parameter Scaling: as task mastery increases, adapter → permanent expert
- This IS the biological memory consolidation analog (hippocampus → cortex)
- The adapter BECOMES part of the model structure

### 2. FlexMoRE — Post-hoc Low-Rank Expert Extraction
- Take a full fine-tuned model, compute Δ = W_finetuned - W_base
- Apply truncated SVD: Δ ≈ U_r Σ_r V_r^T
- This extracts a compact expert that preserves 93-107% of original quality
- 5/6 experts IMPROVED over full fine-tune (SVD acts as regularizer)
- Rank varies by task: knowledge tasks peak at r=4, reasoning at r=2896

### 3. PHATGOOSE — Post-Hoc Tokenwise Expert Routing
- Independently-trained LoRA experts with learned post-hoc routing
- Per-token, per-layer expert selection
- Achieves zero-shot generalization across expert combinations
- This IS Pierre's architecture, independently discovered

### 4. SMEAR — Soft Merging of Experts with Adaptive Routing
- Constructs single merged expert via LEARNED weighted average
- Unlike our fixed W_combined (killed #303), SMEAR's weights are adaptive
- Enables standard gradient-based training (no discrete routing issues)

### 5. MINGLE — Null-Space Constrained Expert Gating
- SVD-extracted experts compose additively (like LoRA)
- But: overlapping parameters cause interference
- Solution: project new expert gating into null space of existing experts
- This IS our Grassmannian approach, applied to solidified experts

### 6. FuseChat — Cross-Architecture Model Fusion
- Fuses models from DIFFERENT architectures (Mixtral + Qwen + InternLM)
- Two-stage: token alignment → parameter space merge
- ONLY paper that truly works without a shared base

### 7. CAMEL — Autonomous Expert Tuner
- Detects concept drift → instantiates new expert
- Freezes old experts to prevent forgetting
- Prunes underused experts
- This IS the adapter promotion/flywheel concept

## The Fatal Flaw We Almost Made

**You CANNOT remove the base model.** Here's why:

```
expert_i = SVD(W_merged - W_base)  → this is a DELTA, not a model
y = Σ w_i * expert_i(x)           → sum of deltas WITHOUT base = noise
```

The SVD-extracted expert only encodes what CHANGED from the base.
Without the base, the foundational knowledge is gone. FlexMoRE explicitly
requires: W_final = W_base + Σ gated_delta_i

## The Corrected Architecture

The base model IS needed — but it should be MINIMAL and FROZEN.
The experts carry the intelligence. The base provides the scaffold.

```
Phase 1: TRAIN adapter on base (what we do now)
Phase 2: SOLIDIFY — SVD extract delta into compact expert
  Δ_i = W_merged - W_base
  expert_i = TruncatedSVD(Δ_i, rank=r_i)   # rank varies by task complexity
Phase 3: COMPOSE — base + routed solidified experts
  y = base(x) + Σ g_i(x) * expert_i(x)     # PHATGOOSE-style gating
Phase 4: GROW — freeze universal experts into extended base
  base_new = base + promoted_experts          # base grows, never retrains
```

### What's different from current Pierre:
- Experts are SVD-extracted (compact, regularized) not raw LoRA
- Rank is heterogeneous (r=4 for knowledge, r=64+ for reasoning)
- Gating is LEARNED per-token (PHATGOOSE) not fixed ridge regression
- Null-space constraint (MINGLE) replaces Grassmannian A-matrix
- Base grows through promotion (CAMEL/ScaleZero), never retrained

### What stays the same:
- Base is frozen during serving
- Experts compose additively in activation space (not weight space)
- Orthogonality prevents interference (null-space = Grassmannian generalization)
- Per-token routing (proven at 98.3% accuracy, Finding #310)

## The Key Insight

The adapter-base fight at scale=20 is NOT a bug — it's telling us the adapter
has OUTGROWN the low-rank constraint. The solution isn't to shrink the adapter
(scale=1-5) or remove the base. It's to:

1. Let the adapter grow to full rank during training
2. SVD-extract the essential directions (FlexMoRE: rank varies by task)
3. The extracted expert is MORE efficient than raw LoRA (regularized)
4. Compose experts via learned gating (SMEAR/PHATGOOSE)

This is adapter solidification: temporary adaptation → permanent expertise.

## Experiments Needed

1. SVD extraction quality: train LoRA at scale=20, merge, extract SVD at various ranks
   Measure: does extracted expert match or beat raw LoRA quality?

2. Rank sensitivity: what rank does each domain need?
   (FlexMoRE: knowledge=4, reasoning=2896. Verify on our domains.)

3. Composition of SVD experts: do they compose better than raw LoRA?
   (MINGLE shows null-space gating helps. Test with Grassmannian.)

4. Scale calibration: at what SVD rank does composition stop degrading MMLU?
   (This answers the scale=20 catastrophe question directly.)

5. Promotion test: freeze one expert into base, add more on top.
   (CAMEL/ScaleZero show this works. Verify on our system.)
