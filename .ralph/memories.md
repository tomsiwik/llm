# Research Memories

### mem-seed-001
> Merging always dilutes expert quality. Routing beats merging — runtime LoRA serving (no merge) preserves both domains at full quality.
<!-- type: finding | tags: merging, routing | source: experiment finding-list --status conclusive -->

### mem-seed-002
> LoRA deltas are naturally orthogonal across domains (cosine ≈ 0.001, 40x below threshold). Grassmannian init guarantees this structurally.
<!-- type: finding | tags: orthogonality, lora | source: 23 BitNet experiments -->

### mem-seed-003
> Use the `experiment` CLI for all experiment state management. Commands: list, get, update, add, evidence, query, stats, refs, finding-add, finding-list, finding-get, finding-update. Never edit HYPOTHESES.yml or FINDINGS.md directly — use the CLI.
<!-- type: pattern | tags: workflow, cli | source: project convention -->

### mem-seed-004
> PROOF-FIRST RESEARCH (Constructive Mathematics). Three experiment types: (1) Verification — complete proof, experiment confirms predictions; (2) Guided Exploration — proven framework, discover unknown parameters within it; (3) Frontier Extension — extend proven math into new territory, find the gap. All three require a mathematical framework. "Try X and see" is never valid.
<!-- type: pattern | tags: methodology, constructive-math | source: 2026-03-28 research reframe -->

### mem-seed-005
> SIGREG REASONING CHAIN (apply to every hypothesis): (a) Treating symptoms or disease? If adding 3rd+ fix, STOP — find single constraint. (b) Reframe: not "prevent X" but "what optimal structure makes X impossible?" (c) Answer from EXISTING math (JL-lemma, Welch, Cramer-Wold, contractions). Hard part is the question, not the technique. (d) Each eliminated hyperparameter = one understood degree of freedom. Ref: LeJEPA 2511.08544, LeWorldModel 2603.19312.
<!-- type: pattern | tags: methodology, sigreg | source: 2026-03-28 LeCun research analysis -->
