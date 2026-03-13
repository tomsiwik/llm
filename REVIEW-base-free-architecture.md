# Peer Review: Base-Free Composable Model Architecture

## Summary of Proposal

Replace a single pretrained model W with a sum of N rank-r components:
W = A_1 B_1 + A_2 B_2 + ... + A_n B_n, each independently trainable.
Bootstrap by SVD of pretrained weights, then selectively retrain components on domain data.

---

## Mathematical Soundness

### 1. The SVD Bootstrapping Is Not What You Think

The proposal says: "SVD its weight matrices into rank-8 components. Each becomes an expert."

This is mathematically well-defined but semantically misleading. SVD of W gives:

W = U Sigma V^T = sum_{i=1}^{d} sigma_i u_i v_i^T

Grouping into rank-8 blocks: W = sum_{k=1}^{d/8} A_k B_k where each A_k B_k captures 8 consecutive singular values.

**The problem:** These are not "domain experts." They are ordered by variance explained, not by semantic content. The first block (sigma_1 through sigma_8) captures the highest-variance directions of W -- likely general language structure, not "JavaScript" or "humor." The last block captures noise.

**Retraining doesn't fix this cleanly.** If you retrain A_k B_k on JavaScript data, you are overwriting whatever variance those 8 singular directions originally captured. The other blocks that depended on those directions being present in W will now see corrupted input. The sum W' = A_1' B_1' + ... (with A_k' retrained) is NOT guaranteed to be close to a good model for non-JavaScript tasks.

**The fix:** Don't bootstrap from SVD at all. Keep the full pretrained W frozen as a base, and ADD independent rank-8 LoRA deltas: W_eff = W_base + sum A_k B_k. This is... exactly what VISION.md already describes. The "base-free" framing is a distraction from the architecture you've already validated.

### 2. The Rank Arithmetic Is Misleading

**Claim:** "For d=4096 at rank-8, you can have d^2/r^2 = 262,144 orthogonal experts."

This bound comes from random subspace packing in R^D where D = d^2 = 16.7M. The expected cosine between two random rank-r subspaces in R^D is approximately r/sqrt(D), which is indeed small.

**But the bound is vacuous in practice.** You will never need 262,144 domain experts. The real constraint is not geometric capacity but:

(a) **Rank budget.** If you have N experts each rank-8, the composed model has effective rank at most 8N. For N=500, that's rank 4000, which is nearly full rank for d=4096. At that point you've parameterized the entire weight matrix -- you have MORE parameters than a single dense matrix, with no benefit.

(b) **The non-orthogonal pairs problem** (see below). Long before you hit geometric limits, semantic overlap creates non-orthogonal pairs.

### 3. The Composition Theorem Only Works for Orthogonal Components

The key property: if A_i B_i is orthogonal to A_j B_j (in the Frobenius inner product sense), then adding/removing one does not affect the other's contribution.

Your data confirms this works beautifully for most pairs (9/10 at cos < 0.002). But the math-medical pair at 0.70 cosine is not an edge case -- it's revealing the fundamental limit.

**Formal problem:** Let W = sum_{k} A_k B_k. Remove expert j. The new weight is W' = W - A_j B_j. The change in output for input x is:

delta_y = W'x - Wx = -A_j B_j x

If A_j B_j is orthogonal to all other experts, this only removes expert j's contribution. But if expert i has cosine 0.70 with expert j, then A_i B_i and A_j B_j share a substantial subspace. Removing expert j doesn't just remove "medical knowledge" -- it removes 70% of a direction that "math" also uses. The math expert's quality degrades.

**This is not fixable by construction.** Math and medical reasoning genuinely share common mechanisms (logical deduction, structured analysis, probabilistic reasoning). Any faithful representation will capture this overlap. You can't make them orthogonal without making at least one of them worse.

---

## The Math-Medical Problem: This Is the Central Issue

### How Common Will Non-Orthogonal Pairs Be?

Your 5-domain experiment gives 10 pairs, 1 of which is non-orthogonal. That's a 10% collision rate at N=5. But this is NOT the right way to estimate scaling.

The collision rate depends on semantic structure, not random geometry. Consider adding these domains to your current 5: statistics, epidemiology, bioinformatics, logic, formal verification, theorem proving, data science, physics, chemistry, pharmacology.

Every single one of these has substantial overlap with either math, medical, or both. At N=50, I'd estimate 20-40% of pairs will have cosine > 0.1, and 5-10% will have cosine > 0.3. At N=500, you'll have clusters of overlapping domains.

### What Happens to Composition in Clusters?

If domains cluster into groups with high internal cosine similarity, then:

1. **Adding a domain expert to a cluster** partially overwrites other cluster members
2. **Removing a domain expert** partially degrades other cluster members
3. **The "modular addition = modular capability" claim breaks down** within clusters

This doesn't kill the architecture. It means the architecture needs explicit handling of non-orthogonal compositions. Options:

(a) **Orthogonalize during training.** Use InfLoRA-style orthogonality constraints (you already have this reference). Force each new LoRA to project out the subspace spanned by existing LoRAs. Cost: O(Nr^2 d) per training step.

(b) **Accept clusters, route cluster-aware.** Don't pretend math and medical are independent. Train a shared "analytical reasoning" expert and layer domain-specific experts on top.

(c) **Gram-Schmidt at merge time.** Before adding expert k, orthogonalize it against all existing experts: A_k' B_k' = A_k B_k - sum_{i<k} proj(A_k B_k, A_i B_i). This preserves only the novel component. Cost: O(N d r) at merge time, negligible.

Option (c) is the most practical and mathematically clean. It means you only add the UNIQUE contribution of each new expert, automatically handling overlap. The downside: removing expert k now requires recomputing the orthogonalization cascade, which is O(N) work.

---

## Hidden Assumptions

### 1. "Precompute the sum for fast inference"

This claim is correct but eliminates the key benefit of modularity. If W_eff = sum A_k B_k is precomputed into a single dense matrix, you get standard inference speed. But:

- Adding/removing an expert requires recomputing W_eff for ALL weight matrices in the model. For Qwen2.5-7B, that's ~200 weight matrices, each 4096x4096. Total: ~6.7GB of computation. Not instant.
- You lose the ability to do per-token routing (which is what makes MoE powerful). Every token uses every expert.
- If every token uses every expert, why not just train a single model? The only advantage is modular training, which is real but different from the MoE narrative.

### 2. "No base model needed"

The proposal conflates two different things:

(a) **At initialization:** You need a pretrained model to SVD. Without it, you're training from scratch with a specific parameterization (sum of rank-8 matrices), which has no advantage over standard training and many disadvantages (optimization landscape is harder).

(b) **At inference:** After precomputing the sum, there is no "base" in the computation graph. But the pretrained model's knowledge IS the sum. You haven't eliminated the base -- you've distributed it across components.

This is like saying "I don't have a savings account; I have 500 separate $100 bills." The money is still there.

### 3. "Train one expert at a time without affecting others"

This holds only if orthogonality holds. Your own data shows it doesn't always hold. And even when it does hold geometrically, it doesn't guarantee semantic independence.

Consider: Expert A (Python) and Expert B (Bash) are trained independently and are orthogonal. But if a user asks "convert this Python script to Bash," neither expert alone has the cross-domain knowledge. The composed model's behavior depends on cross-expert interaction terms that were never trained.

### 4. Rank-8 is sufficient per expert

Your rank sweep shows rank 8 gives PPL 1.59 vs rank 128 at PPL 1.50 for bash FFN-only. That's a ~6% gap. On a single narrow domain (shell commands), that might be acceptable. On complex domains (legal reasoning, medical diagnosis, creative writing), rank 8 may be catastrophically insufficient.

The diminishing returns argument cuts both ways: yes, going from 8 to 128 gives only 6% improvement. But that 6% could be the difference between "useful expert" and "hallucinating expert" on hard queries.

---

## What IS Sound

1. **LoRA orthogonality is real and well-measured.** cos=0.001 across 9/10 domain pairs is strong evidence. This is the foundation of the architecture and it holds.

2. **Simple averaging dominates for orthogonal LoRAs.** Your merging bakeoff at micro scale is thorough and the conclusion is robust: when components are orthogonal, just add them.

3. **The economics are genuinely compelling.** $0.25/expert, 15 min training, 6MB storage. If the quality holds, this is a legitimate advantage over retraining.

4. **Precomputed sum = standard inference speed.** Mathematically trivial but practically important.

5. **The N_max scaling law is mathematically correct** as a geometric bound. It just isn't the binding constraint.

---

## Alternatives and Better Framings

### 1. Don't call it "base-free." Call it "base-distributed."

The pretrained model IS the sum of your initial SVD components. Retraining individual components makes the model evolve, but the base knowledge persists in the untouched components. This is a feature, not something to hide.

### 2. Keep the frozen base, add LoRA experts (your existing architecture)

W_eff = W_frozen + sum_{active} A_k B_k

This is strictly better than the base-free version because:
- W_frozen captures all the general knowledge at full rank
- Each A_k B_k only needs to capture domain-specific DELTA
- If all LoRAs are removed, you gracefully degrade to base model quality
- Adding a bad LoRA can be undone by simply removing it

The "base-free" version loses this safety net. If you retrain SVD component k badly, you've corrupted the model and can't easily recover (because the other components' behavior assumed the original component k).

### 3. Hierarchical LoRA

Instead of flat sum, use a two-level hierarchy:
- Level 1: Small set of "foundation LoRAs" (rank-64) for broad categories (reasoning, code, language, factual)
- Level 2: Large set of "specialist LoRAs" (rank-8) for specific domains, applied ON TOP of the relevant Level 1

This handles the math-medical problem naturally: both get the "reasoning" Level 1 LoRA, then diverge at Level 2.

---

## Macro-Scale Risks (advisory)

1. **Non-orthogonal cluster growth.** As you add hundreds of real-world domains, clusters of overlapping LoRAs will emerge. Need an explicit strategy (orthogonalization, clustering, hierarchy).

2. **Rank budget exhaustion.** At N=500 with rank-8, total parameter count in LoRAs alone is 500 * 2 * 4096 * 8 = 32.8M per weight matrix. That's ~2x the parameters of the weight matrix itself (4096^2 = 16.8M). You're overparameterized. Consider whether this actually helps vs. a single high-rank adaptation.

3. **Quality verification at scale.** How do you verify that expert #347 hasn't subtly degraded expert #12? At N=500, there are 124,750 pairwise interactions to check.

4. **The "no routing" limitation.** If you precompute the sum, every token pays for every expert. At N=500, this means every token's output is influenced by bash, medical, humor, etc. There's no specialization at inference time. This is the opposite of MoE's efficiency advantage.

---

## Verdict

**REVISE** -- with specific architectural changes.

The core insight (orthogonal LoRAs compose cleanly via addition) is proven and valuable. But the "base-free" framing introduces real mathematical problems that the existing VISION.md architecture already solves. Specific fixes:

1. **Drop the "base-free" framing.** Keep the frozen base model. The math is cleaner, the safety properties are better, and it's what you've already validated. The base model IS a feature.

2. **Add Gram-Schmidt orthogonalization at merge time.** Before adding expert k, project out its component along all existing experts. This handles the math-medical problem by only adding the NOVEL contribution of each expert. Implement and measure: does orthogonalized expert still improve its domain?

3. **Quantify the non-orthogonality problem at scale.** Train 20-50 LoRA experts on semantically diverse domains (not just 5). Measure the full pairwise cosine matrix. Plot the distribution. This is the single most important empirical question: does the 10% collision rate at N=5 grow sublinearly (manageable) or superlinearly (architecture-breaking)?

4. **Address the rank budget question.** At what N does sum of rank-8 LoRAs exceed the parameter count of the weight matrix? Is there evidence of diminishing returns before that point? (Your N=5 merging bakeoff at +3.33% gap suggests yes.)

5. **Decide: precomputed sum vs. per-token routing.** These are fundamentally different architectures with different tradeoffs. The proposal conflates them. Precomputed sum = simple, no routing cost, no specialization per token. Per-token routing = MoE, routing cost, but only activate relevant experts. You need one story, not both.

The honest pitch for this architecture is: "Orthogonal LoRA experts on a frozen base compose cleanly via simple addition. We can add domain knowledge at $0.25/expert with near-zero interference. The system degrades gracefully when experts are removed." That's already novel and valuable. The "base-free" extension adds mathematical risk without commensurate benefit.
