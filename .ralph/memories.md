# Research Memories

### mem-seed-001
> Merging always dilutes expert quality. Routing beats merging — runtime LoRA serving (no merge) preserves both domains at full quality.
<!-- type: finding | tags: merging, routing | source: FINDINGS.md -->

### mem-seed-002
> LoRA deltas are naturally orthogonal across domains (cosine ≈ 0.001, 40x below threshold). Grassmannian init guarantees this structurally.
<!-- type: finding | tags: orthogonality, lora | source: 23 BitNet experiments -->

### mem-seed-003
> Use the `experiment` CLI for all experiment state management. Commands: list, get, update, add, evidence, query, stats, refs. Never edit HYPOTHESES.yml directly.
<!-- type: pattern | tags: workflow, cli | source: project convention -->
