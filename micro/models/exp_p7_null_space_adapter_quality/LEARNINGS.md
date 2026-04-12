# LEARNINGS: P7.A1 — Null-Space Adapter Quality

## Core Finding
Null-space LoRA (A = A_null @ Q^T, arXiv:2512.15233) preserves 98.7% of unrestricted adapter quality while guaranteeing exact orthogonality to W_v (max violation 1.33e-5, 100x below threshold). The null-space restriction is essentially free: same final loss (0.037), same convergence rate, 20% fewer parameters.

## Why It Works
SVD of W_v yields a 2048-dim null space where gradients are unrestricted (Theorem 2: gradient retention ratio d_null/d_in = 0.80 lower bound). The reparameterization A = A_null @ Q^T ensures W_v @ A_eff^T = 0 by construction — no learned penalty, no regularization needed.

## Critical Discovery: Gemma 4 KV-Sharing
Layers 24-41 of Gemma 4 E4B receive pre-computed KV from layers 22/23 via `shared_kv`. v_proj is **dead code** on those layers — adapters have zero effect. **Any future experiment targeting k_proj/v_proj on Gemma 4 must verify layers 16-23 (non-shared) only.**

## Implications for Next Experiment
P7.A2 (two null-space adapters on the same layer) is now unblocked. Capacity is ample: 2048 null dims / r=16 = 128 non-overlapping slots per layer, covering 25+ domains with 5x headroom. The remaining question is whether orthogonality in weight space translates to functional independence in activation space.

## Caveats
- Memorization scale (20 texts, PPL=1.03) — ratio may differ on harder tasks at larger data scale
- K1298 vacuous (base PPL=8154); null-space general PPL (362) is 44.7% worse than unrestricted (250) — relevant for composition
- P2 (post-hoc projection) untested; separate experiment if needed
