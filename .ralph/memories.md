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

### mem-p1-001
> PIERRE P1: Base model is Gemma 4 on MLX. 40 experiments tagged `p1` in experiment DB (`experiment list -t p1`). 7 tiers: T0=math foundation, T1=orthogonality bakeoff, T2=adapter training, T3=composition, T4=routing+serving, T5=user pipeline, T6=crystallization. Start with T0 (`experiment list -t p1 -t t0-foundation`).
<!-- type: project | tags: p1, gemma4, mlx | source: 2026-04-09 P1 design session -->

### mem-p1-002
> MLX GEMMA 4 GUIDE: Read `docs/MLX_GEMMA4_GUIDE.md` BEFORE writing any Gemma 4 code. Contains: verified model IDs (use `mlx-community/gemma-4-e4b-it-4bit` for dev, `mlx-community/gemma-4-26b-a4b-it-4bit` for production), exact LoRA training commands, GrassmannianLoRALinear implementation (section 4), Gemma 4 layer map (section 5), config dimensions (section 6), known issues (section 9).
<!-- type: reference | tags: mlx, gemma4, guide | source: 2026-04-09 docs/MLX_GEMMA4_GUIDE.md -->

### mem-p1-003
> GRASSMANNIAN ADAPTER ON MLX: Replace `lora_a` in mlx-lm's `LoRALinear.__init__` with pre-computed QR slot. The A-matrix is frozen (not in trainable params), only B is trained. Code: `self._grassmannian_a = mx.array(slots[domain_id])` then forward: `z = (x @ self._grassmannian_a) @ self.lora_b`. Orthogonality guarantee: `trace(... Y_j^T Y_i) = 0` because QR construction gives `Y_i^T Y_j = 0`. Full impl in `docs/MLX_GEMMA4_GUIDE.md` section 4.
<!-- type: pattern | tags: grassmannian, lora, mlx | source: 2026-04-09 P1 adapter design -->

### mem-p1-004
> GEMMA 4 ARCHITECTURE FOR ADAPTERS: 30 layers (26B-A4B): 25 sliding (local, head_dim=256, standard RoPE) + 5 global (full_attention, head_dim=512, K=V, p-RoPE 25%). Domain adapters go on LOCAL layers (q_proj only). Global layers have K=V so V is shared — adapters on global only modify Q. PLE injection point exists after attention+FFN (gated residual). V-norm (RMSNorm without learned scale) is already in Gemma 4 — prevents scale catastrophe. Full math: `../ARCHITECTURE_P1.md`.
<!-- type: reference | tags: gemma4, architecture, adapters | source: 2026-04-09 ARCHITECTURE_P1.md -->

### mem-p1-005
> ORTHOGONALITY ALGORITHMS (UNEXPLORED): Householder Reflections (HRA, arXiv:2405.17484) — cheapest: O(rd) params, exact orthogonality by construction, beats LoRA by +5.6pp GSM8K at half params. Givens Rotations (qGOFT, arXiv:2404.04316) — O(d) params, parallelizable. Cayley Transform (arXiv:2002.01113) — O(r^2), exact. PoLAR Landing Field (arXiv:2506.03133) — approximate, but full stable rank. Comparison table in `../ARCHITECTURE_P1.md` section 3.
<!-- type: reference | tags: orthogonality, householder, polar | source: 2026-04-09 ARCHITECTURE_P1.md section 3 -->

### mem-p1-006
> IMPOSSIBILITY STRUCTURES (why P1 math can't fail): (1) Parameter interference=0: QR gives Y_i^T Y_j=0, trace with zero=zero, algebraic not statistical. (2) Scale safety: V-norm forces ||V||_RMS=sqrt(d) regardless of adapter magnitude. (3) Plug-and-play: composition is additive W+sum(delta_i), adding/removing one doesn't touch others. (4) Quantization-safe: only B quantized, Grassmannian Y frozen, interference stays zero. (5) User isolation: orthogonal subspaces, no info crosses.
<!-- type: finding | tags: impossibility, composition, p1 | source: 2026-04-09 P1 experiment design -->
