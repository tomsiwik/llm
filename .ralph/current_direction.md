# Current Direction: Gumbel-Sigmoid Routing Ablation

**Experiment**: exp_gumbel_sigmoid_ablation
**Status**: active
**Started**: 2026-03-26

## What
Systematic ablation of the Gumbel-sigmoid router configuration used in N=50 composition.
Testing temperature, top-k, competing vs non-competing gates, load-balancing loss,
and straight-through estimation. Special focus on the 4 zero-accuracy domains
(chemistry, wikitext, dialogue, debate).

## Baseline
From bitnet_scale_n50:
- Router: 2-layer MLP (d -> 128 -> N), Gumbel-sigmoid
- Temperature: annealed 2.0 -> 0.5 over 600 steps
- Top-k: 2, accuracy 86.33%
- 4/49 domains at 0% accuracy
- gamma_routed = 0.632

## Kill criterion
K1: No configuration beats current default (86.33% top-2 accuracy) by >5% (i.e., >91.33%).

## Approach
Reuse N=50 trained adapters and data. Only retrain the router under different configs.
Extract hidden states once, then sweep configs cheaply.
