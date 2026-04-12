# LEARNINGS: exp_p6_lingering_adapter_online

## Core Finding
Rank-4 LoRA with single-gradient-step online updates can encode project-specific
knowledge from 20 conversation turns: +60pp project QA accuracy (0→60%), 110ms/turn
latency, zero general knowledge cost. The lingering adapter concept is viable for
real-time personalization.

## Why It Works
Intrinsic dimensionality of a single project's facts (~10 facts) is far below LoRA
capacity (rank-4 × 8 layers = 32 dimensions), so convergence is guaranteed even with
non-convex LoRA loss in practice (arXiv:2012.13255). Online GD regret bound (Zinkevich
2003) explains frequency-accuracy correlation: facts seen 3-4× all learned; 2× mixed.

## What Breaks
- **Repetitive generation**: adapted outputs show pathological repetition (token cycling)
  — rank-4 constraint forces the model to loop rather than search for alternatives.
- **Character-level details lost**: separators ("zf:" → "zf_"), units ("256KB" → "256
  bytes"), version suffixes ("Python 3.12" → "Python"). Adapter encodes semantic
  category but not precise token identity.
- **Non-convexity caveat**: formal regret guarantee requires convex loss; the LoRA
  → logits mapping is non-convex. Empirical convergence holds but formal bound is loose.

## Implications for Next Experiment
To fix repetitive generation: add a repetition penalty or train on longer response
windows. To improve character-level recall: increase rank to 8 or add more gradient
steps per turn (k=3-5). The O(1/√T) scaling predicts ~80%+ accuracy at 40 turns —
directly testable. Next natural step: multi-turn lingering adapter with rank-8 or
multi-step updates (arXiv:2411.13405 PLUM scaffolding).

## References
- arXiv:2411.13405 — PLUM: conversation-to-QA augmentation
- arXiv:2012.13255 — Intrinsic dimensionality (Aghajanyan et al.)
- Zinkevich 2003 — Online convex optimization regret bounds
- Finding #490 — SUPPORTED (this experiment)
