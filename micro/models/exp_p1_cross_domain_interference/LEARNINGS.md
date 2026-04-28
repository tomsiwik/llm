# Learnings — exp_p1_cross_domain_interference

## Core Finding
Cross-domain interference is real, asymmetric, and pattern-dependent. Adapters that learn broad general features (python, medical) cause 12-14pp degradation off-domain, while the narrower math adapter is benign (+2, +8pp). Surprising positive transfers exist (python→MedQA +50pp).

## Why
Each adapter reshapes the residual stream differently. Broad adapters (python learned general reasoning, not just syntax) overwrite features other domains need. Narrow adapters (math = numeric patterns) occupy orthogonal subspaces and don't collide. The asymmetry means interference is structural, not uniform noise.

## Implication for Next Experiment
Soft weighted routing is required — hard routing loses to uniform (Finding #826), but uniform suffers from interference (Finding #827). The router must learn per-sample blend weights that suppress harmful adapters (python on math, medical on code) while preserving beneficial transfers. This is the gating/MoE-style experiment.
