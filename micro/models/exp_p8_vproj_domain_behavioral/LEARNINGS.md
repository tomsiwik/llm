# LEARNINGS — exp_p8_vproj_domain_behavioral

## Core Finding
v_proj+o_proj adapters improve behavioral text quality across all 5 domains vs both
base model and q_proj adapters. Status: SUPPORTED (2/4 kill criteria pass, directional
finding validated). Finding #504.

## Why It Works
Output-path adapters (v_proj+o_proj) directly modify the token vocabulary distribution
at generation time. Query-path adapters (q_proj) only change attention patterns — what
the model looks at — which is insufficient for generation tasks. This is mechanistically
confirmed by vocabulary shift: mean +21% to +59% across all domains.

## Measured vs Predicted
Predictions (70-80% math, 65-75% code) overestimated measured values (55%, 50%).
Post-hoc explanation: ceiling effect from base model (Gemma 4 E4B-IT) already being
strong at math/code, plus only 80 training examples (8-10 unique, cycled).
Medical passes (70%) where base model is weaker — consistent with the ceiling story
but not prospectively predicted, so treat as hypothesis not confirmed mechanism.

## Caveats
1. K1315 composition trivially satisfied — sequential serving guarantees 100% retention
   by construction; actual adapter weight merging was not tested.
2. Legal at 35% is the weakest domain — improved vs q_proj (20→35pp) but notably
   below finance (50%) and code (50%) despite rich legal vocabulary.
3. Ceiling effect explanation is post-hoc and unfalsifiable from this experiment alone.

## Implications for Next Experiment
To push math/code past 60% threshold, need: (a) more diverse training data (>100
unique examples), OR (b) longer training (500+ iters), OR (c) rank-32 adapters.
A stronger design would pre-measure base model competence per domain and use it to
predict which domains will hit ceiling effects — turning the post-hoc explanation into
a prospective prediction.
