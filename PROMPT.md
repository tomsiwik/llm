Composable Ternary Experts — Research Loop on Apple Silicon (M5 Pro 48GB, MLX only).

## What We're Building
Ternary base (BitNet-2B-4T) + composable adapters + structural interference guarantees.
Goal: adding a new domain costs $2 and 10 minutes, not $10K and a week.

## What We Have
- 5 real-data adapters (medical/code/math/legal/finance, -26.3% PPL)
- Near-lossless composition at N=5 (Finding #225)
- Gumbel-sigmoid routing (44% better than softmax)
- Pre-merge serving (0% overhead on MLX, 165 tok/s)
- Fully ternary adapters (15.8x compression)

## Proof-First Method (no exceptions)
a) Symptom or disease? If 3rd+ fix → STOP, find root cause
b) Reframe: "what structure makes failure impossible?" not "how to prevent X?"
c) Derive from existing math (JL-lemma, Welch, concentration inequalities)
d) MATH.md: Theorem/Proof/QED with quantitative + behavioral predictions
e) Code: verify predictions on MLX
f) PAPER.md: prediction-vs-measurement table
Types: verification | guided-exploration | frontier-extension
Ref: LeJEPA (2511.08544), SIGReg method

## Priorities
P0: Ship working system — generation quality, benchmarks, 25 domains, e2e pipeline
P1: Composition scaling �� null-space isolation, Fisher-Rao merging, ridge regression routing
P1: Train own ternary base — STE, Falcon-Edge toolkit, Sparse-BitNet
P2: Production serving — adapter hot-swap, per-token routing, caching

## Each Iteration
1. `experiment claim <worker>` — pick highest-priority unblocked work
2. Read kill criteria + notes from claim output
3. Invoke `/fast-mlx` `/mlx-dev` before MLX code
4. Write MATH.md → implement → run �� write PAPER.md
5. `experiment complete <id> --status supported --dir micro/models/<name>/ --k <id>:pass --evidence "..."`
6. If no open experiments: generate hypotheses from `experiment finding-list` + `experiment query`
7. Only output RESEARCH_BACKLOG_DRAINED when nothing actionable remains

## Orphan Check (do FIRST)
Read .ralph/current_direction.md. If last experiment's REVIEW-adversarial.md or LEARNINGS.md missing → resolve before new work.

## Rules
- Each experiment <2hrs. If stuck, wrap partial results and move on
- ALL experiments on MLX/Apple Silicon. No CUDA
- `uv run` for Python. `experiment` CLI for all state management
- Every new hypothesis MUST cite arxiv paper or prior finding. No analogies
- Killed experiments: derive impossibility structure, then re-test
- KEEP GOING. After each cycle, pick next experiment. Never stop early
