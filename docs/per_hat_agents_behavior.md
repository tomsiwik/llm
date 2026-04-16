# Per-hat AGENTS behavior in Ralph

Checked against official Ralph docs:
- `guide/configuration/`
- `guide/cli-reference/`
- `guide/prompts/`
- `guide/backends/`
- `concepts/hats-and-events/`
- `advanced/custom-hats/`
- `api/config/`

## Short answer

No: the Ralph docs do **not** describe a built-in mechanism like "one `AGENTS.md` per hat".

What Ralph documents instead is:
- one global `prompt_file` for the loop
- global `core.guardrails`
- per-hat `instructions`
- per-hat `backend`
- per-hat backend-specific args
- split config support via `-c` (core) and `-H` (hat collection)

So Ralph's native abstraction for hat-specific behavior is **hat instructions**, not per-hat `AGENTS.md` discovery.

## What the docs explicitly support

### Global prompt
Ralph supports a loop-level prompt file:
- `event_loop.prompt_file: "PROMPT.md"`

That is global to the run, not per hat.

### Per-hat instructions
Hat config supports role-local instructions:
- `hats.<id>.instructions`

This is the documented place for hat-specific behavior.

### Per-hat backend override
Hat config also supports per-hat backend choice:
- `hats.<id>.backend`

### Per-hat backend args
The config/API shape supports per-hat backend args.
In this repo you are already using that pattern with Claude backend args.

## What is NOT documented as a Ralph feature

The docs do not show any native feature like:
- `hats.<id>.prompt_file`
- `hats.<id>.agents_file`
- automatic per-hat `AGENTS.md` discovery
- automatic role-based prompt file switching by hat

So if you want per-hat AGENTS-like behavior, that is something you must implement **on top of Ralph**, not something Ralph itself provides.

## Practical ways to get per-hat behavior

### Option A — Recommended: thin `ralph.yml`, role files referenced from hat instructions
Use:
- repo-global `AGENTS.md` for universal rules
- skills for command-heavy operational knowledge
- per-hat files like:
  - `ralph/hats/researcher.md`
  - `ralph/hats/reviewer.md`
  - `ralph/hats/analyst.md`

Then each hat instruction says to read its own file.

This is Ralph-compatible because the hat-specific behavior still lives under documented `instructions`.

### Option B — Backend-specific prompt injection
For Claude specifically, you can pass backend args such as appended system prompt files per hat.
That works, but it is **backend-specific**, not a generic Ralph behavior.

### Option C — Split hat collections
You can separate core config and hat collections using:
- `-c` for core
- `-H` for hats

This helps organization, but still does not create per-hat `AGENTS.md` semantics by itself.

## Best conclusion

If your goal is "separate AGENTS-like behavior per hat," the docs support this pattern best:
- keep one repo-global `AGENTS.md`
- move domain procedure into skills
- give each hat its own small role file and reference it from `instructions`
- optionally use backend-specific system prompt files where useful

That achieves per-hat specialization without pretending Ralph has a native per-hat `AGENTS.md` feature.
