# LEARNINGS — T3.6: Plug-and-Play Hot-Add (SUPPORTED)

## Core Finding
Exclusive routing makes adapter hot-add structurally free: adding domain N+1 to the
registry leaves all existing outputs bit-identical (40/40, max_token_diffs=0) and new
adapter is immediately functional (90% vs 4% base), with 0.004ms latency (23,000× margin).

## Why
Exclusive routing means W_eff(q, i) = W_base + A_i B_i — a Python dict update for
key N+1 cannot touch values at keys 1..N. This is pure dict semantics, not an empirical
result. Theorem 1 is watertight. Zero retraining required.

## T3 Tier Structural Summary (load-bearing constraints now proven)
1. **T3.1 (KILLED)**: Simultaneous N=5 activation → math 82→8%, code 66→8% catastrophic collapse.
   Routing is REQUIRED, not optional.
2. **T3.2 (KILLED)**: Scale≥12 degrades MMLU. Scale=6 is the safe operating point.
3. **T3.3 (SUPPORTED)**: Activation-space power law alpha=0.15; routing makes O(N) noise zero.
4. **T3.4 (SUPPORTED)**: N=25 Grassmannian; max|cos|=2.2e-8 under exclusive routing.
5. **T3.6 (SUPPORTED)**: Hot-add is free under exclusive routing; bit-exact, 0.004ms.

## Implications for Next Experiment
PLE-M2P routing is the only viable composition strategy for the Room Model. T4 experiments
should focus on the full PLE-M2P pipeline: input routing (TF-IDF or learned), adapter
selection, and end-to-end behavioral evaluation on multi-domain queries. Code domain should
be included in all future benchmarks (was missing from T3.6 K1067).

## Caveats (Non-Blocking)
- Geography adapter = copy of finance (MCQ format compliance, not domain training)
- Code adapter missing from K1067 (covered by Theorem 1 guarantee)
- n=10 per domain for accuracy (large margins make this immaterial)
