# Current Direction

## Completed: exp_embedding_routing_n24 -- KILLED
**Status:** Killed (K590 FAIL, K591 FAIL, K592 PASS)
**Completed:** 2026-03-29

## Key Result
Embedding-layer routing is WORSE than hidden-state routing (25.2% vs 32.5%).
Centroid collapse: embedding centroids have mean cosine 0.986 (nearly identical).
Transformer layers ADD discriminative signal (hidden cos 0.716), not destroy it.
TF-IDF (35.0%) beats neural embeddings because IDF downweights common words.

Sixth routing kill at N=24. The ~40% ceiling is representation-quality-limited.

## Findings
- #194: Embedding centroid collapse from shared vocabulary
- #195: Transformer layers ADD discriminative signal
- #196: TF-IDF outperforms neural embedding routing

## Implications for Routing at N=24
The problem is NOT which layer to extract features from. It is that 24 overlapping
text domains cannot be separated by any simple representation without a specialized
discriminative model. Options:
1. External sentence encoder (LoRAuter approach -- needs second model)
2. Accept ~40% accuracy (already proven PPL-benign at oracle gamma = softmax gamma)
3. Hierarchical clustering to reduce effective N
4. Per-token routing with mixed-domain data
