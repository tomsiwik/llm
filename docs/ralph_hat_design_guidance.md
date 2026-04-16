# Ralph hat design guidance for this repo

Based on official Ralph docs:
- `concepts/hats-and-events/`
- `concepts/coordination-patterns/`
- `concepts/memories-and-tasks/`
- `guide/prompts/`
- `guide/agents/`

## What Ralph expects

Ralph's model is event-driven and hat-centric:
- each hat has a **single responsibility**
- hats communicate with **small typed events**
- event payloads are for routing and compact handoff, not large state transfer
- persistent context should live in **memories/tasks/files**, not in giant event payloads
- triggers should be **specific**, not wildcard-heavy
- hat collections define workflow topology; Ralph stays the constant coordinator

## Practical implications for this repo

### 1. Keep experiment knowledge out of the Ralph orchestration contract
The `experiment` CLI is domain-specific workflow knowledge.
That belongs in:
- `AGENTS.md`
- experiment-related skills
- hat-specific prompt/instruction files if a given hat needs it

It does **not** belong in Ralph's generic orchestration model beyond minimal references like:
- what event starts the loop
- what hats exist
- which hat publishes which event

### 2. Design hats around role boundaries, not project policy bundles
Per the docs, good hats are focused specialists. In this repo that means:
- **researcher**: claim/run/document experiment
- **reviewer**: adversarially validate evidence and docs
- **analyst**: distill learnings and implications

Each hat should know only the policy needed for its role.
Do not make every hat ingest the entire experimentation handbook if only one role needs part of it.

### 3. Prefer hat-specific instruction files over one giant shared instruction blob
The docs emphasize role-specific instructions and prompt clarity.
So the right shape is:
- core repo-wide rules in `AGENTS.md`
- reusable CLI syntax in skills
- per-hat operating instructions in hat-specific files or in the hat's `instructions`

That keeps Ralph orchestration agnostic while still making each hat competent.

### 4. Use event payloads as the primary handoff signal
This follows Ralph's event-first design.
A good event payload should contain compact routing info such as:
- experiment id
- experiment dir
- status summary
- whether results exist

A bad event payload is full logs, long tables, or whole paper text.

### 5. Use files and memories for durable context
Ralph docs say memories/tasks replace scratchpad-heavy workflows.
For this repo, durable state should prefer:
- experiment DB (`experiment` CLI)
- experiment-local artifacts (`MATH.md`, `PAPER.md`, `results.json`)
- `.ralph/memories.md` for compact cross-loop learnings

Do not rely on a mutable summary file as the only source of truth for hat routing.

## Recommended structure

### AGENTS.md
Keep only repo-global rules and the experiment workflow contract:
- use `experiment` CLI for experiment state
- proof-first methodology
- Apple Silicon / MLX constraints
- artifact expectations (`MATH.md`, `PAPER.md`, etc.)

### Skills
Put command-heavy operational knowledge in skills:
- exact `experiment` CLI syntax
- MLX coding constraints
- specialized evaluation or literature tools

### Ralph hat config
Keep `ralph.yml` mostly about:
- hats
- triggers
- publishes
- backend choice
- loop limits
- minimal role instructions

### Optional next step
If this repo keeps growing, split hat instructions into files:
- `ralph/hats/researcher.md`
- `ralph/hats/reviewer.md`
- `ralph/hats/analyst.md`

Then `ralph.yml` can stay thin and orchestration-focused.

## Bottom line

What the Ralph docs support is:
- **AGENTS/skills own domain knowledge**
- **hats own narrow role behavior**
- **events own routing**
- **Ralph owns coordination**

So your instinct is correct: the experimentation handbook should be consolidated into `AGENTS.md` + skills, while the Ralph loop itself should stay relatively agnostic and role-oriented.
