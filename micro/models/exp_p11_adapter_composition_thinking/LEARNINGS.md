# LEARNINGS.md — P11.J0: Adapter Composition via Exclusive Routing

## Core Finding

Embedding-based routing (cosine similarity on embed_tokens mean-pool) separates reasoning vs
knowledge queries with predicted ≥85% accuracy (K1528), but domain adapters trained with
thinking=False on NTP objectives actively degrade MCQ performance (~26pp, Finding #517),
making K1527 (routed_knowledge ≥ thinking + 2pp) pre-registered to FAIL.

## Why

Room Model exclusive routing guarantees zero cross-adapter interference by construction
(Theorem 3: disjoint adapter application → W_combined = W_base + ΔW_selected, no blending).
The routing mechanism is sound; the bottleneck is adapter quality, not composition mechanics.
Domain adapters trained on q_proj-only with thinking=False learn answer-token distributions,
not the reasoning chains that improve MCQ accuracy.

## Implications for Next Experiment

The pre-registered failure of K1527 directly motivates P11.L0 (RSD-aligned traces):
train domain adapters with thinking=True on MCQ-format data so that routing to a domain
adapter adds value instead of subtracting it. K1528 PASS would confirm the routing
infrastructure is ready; the missing piece is adapter quality.
