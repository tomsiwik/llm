# Current Direction

## Active Experiment
**exp_cross_adapter_knowledge_transfer** -- Cross-domain constructive transfer matrix

## What
Build the full NxN transfer matrix for 5 domain adapters on BitNet-2B-4T.
For each pair (A, B), measure whether adding adapter A improves domain B's PPL.
This maps the knowledge graph of our adapter pool and reveals constructive transfer structure.

## Why
OSRM showed composition works via constructive transfer, not orthogonality.
The composed 5-adapter PPL beats naive 1/N prediction on all domains.
Per-token routing shows confused domains have complementary adapters.
But we have never measured PAIRWISE transfer directly.

## Kill Criteria
- K1 (id=243): Zero pairs show >2% cross-domain improvement
- K2: Transfer matrix is random (no structure, MI < 0.3)

## Status
ACTIVE -- retraining 5 domain adapters, then building full transfer matrix
