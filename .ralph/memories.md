# Research Memories

### mem-seed-001
> LoRA A matrices do NOT self-route (~50% accuracy, coin flip). Routing and computation must be decoupled. Use separate contrastive routing keys K_i trained with InfoNCE loss on ~50 samples in ~50 steps.
<!-- type: fix | tags: routing, lora | source: Exp 1 self-routing validation -->

### mem-seed-002
> Merging always dilutes expert quality. Task arithmetic at λ=0.5 halves expert strength. TIES/DARE/SVD all lose specialization. Routing beats merging — CAT and learned Router preserve both domains at full quality.
<!-- type: finding | tags: merging, routing | source: FINDINGS.md -->

### mem-seed-003
> Capsule MoE (rank-1 decomposition + two-level group routing) matches dense GPT at parameter parity within 0.7%. Lowest variance across seeds. But FLOP savings are theoretical only — conditional execution not yet implemented.
<!-- type: finding | tags: capsule, sparsity, parity | source: capsule_moe experiment -->

### mem-seed-004
> Expert lifecycle (freeze best / recycle worst by weight norm) gives modest improvement (~0.5% less forgetting) when N ≥ M. Starves later domains when N < M. Weight norm is a reasonable proxy for specialization.
<!-- type: finding | tags: lifecycle, forgetting, capacity | source: moe_freeze experiment -->

### mem-seed-005
> MoE routing is the ONLY beneficial mechanism from the 32-combination ablation study. ART-modulated LR hurts (+0.028), Bloom filter zero effect, Splay cache zero effect, ART spawn hurts (+0.005). "Cognitive stack as routing optimizer" narrative is dead.
<!-- type: decision | tags: ablation, pivot | source: FINDINGS.md -->

### mem-seed-006
> LoRA deltas are naturally orthogonal across domains (Python vs JavaScript adapters at cosine ≈ 0.000). This is a structural property, not a coincidence — exploit it for routing.
<!-- type: finding | tags: orthogonality, lora | source: VISION.md -->

### mem-seed-007
> Micro scale constraints: d=64, ~200K params, character-level names dataset, block_size=32. Models use RMSNorm, no bias, ReLU, pre-norm, learned positional embeddings, untied embeddings. Results are directional, not definitive.
<!-- type: context | tags: micro, constraints | source: project convention -->

### mem-seed-008
> Arena model registry: @register("name", parent="parent"). Run via run_single() or run_multidomain(). Multi-domain uses a-m vs n-z split, 300 steps/domain. Always compare against parent baseline.
<!-- type: pattern | tags: arena, implementation | source: micro/arena.py -->

### mem-seed-009
> Next frontier from VISION.md: (1) Contrastive routing keys >85% accuracy, (2) Sparse top-1 routing matching CAT at 1/N compute, (3) Procrustes decomposition for shared/unique separation, (4) Scale to 5+ languages, (5) Beat 1.5B monolithic with 0.5B + experts.
<!-- type: context | tags: roadmap, next-steps | source: VISION.md "What Remains" -->
