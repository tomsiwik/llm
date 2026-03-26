# Current Direction

## Active Experiment
**exp_generation_quality_test** -- Does routed composition produce better TEXT than base alone?

## What
Load BitNet-2B-4T + 5 trained domain adapters + routing. Generate text for
10 prompts per domain (50 total). Compare base-only vs uniform-composition vs
routed-top-2. Score with automated metrics: domain keyword density, response
length, n-gram diversity, and cross-PPL (adapter predicting generated text).

## Why
This is the EXISTENTIAL test for BitNet-SOLE. PPL improvements are proven but
nobody has checked whether the generated text is actually better. Cross-adapter
transfer KILLED (0/20 pairs via blending), confirming routing >> blending.
Now we test whether routing produces real value in generated text.

## Kill Criteria
- K1 (272): Routed worse than base on >= 3/5 domains
- K2 (273): No measurable difference (decorative composition)
- K3 (274): All text incoherent (base too weak)

## Status
ACTIVE -- implementing generation + automated scoring pipeline
