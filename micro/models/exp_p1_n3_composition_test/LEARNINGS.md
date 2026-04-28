# LEARNINGS — exp_p1_n3_composition_test

## Core Finding

Uniform LoRA composition via Σ(A_i @ B_i)/N is non-tautological and cross-domain clean, but degrades accuracy beyond tolerance (math -10pp, code -12pp). The composition mechanism works — uniform weighting does not.

## Why

Dividing total lora_scale equally across N adapters dilutes each adapter's contribution. Domains with sharper weight distributions (math, code) suffer more than diffuse ones (medical, -2pp). The problem is allocation, not interference: cross-domain leakage is only 14% (math on MedQA).

## Implication for Next Experiment

Per-sample routing is necessary. Since interference is low and composition is mechanically sound, the bottleneck is selecting the right adapter(s) per input — not combining them. The routing experiment (exp_p1_n3_routing_accuracy) should test whether a lightweight classifier can recover the single-adapter baselines by routing each sample to its domain adapter.
