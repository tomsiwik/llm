# Task: drain the research backlog

## Objective
Work through open experiments in the DB until no open or active experiments remain at priority ≤ 2.

## Success criteria (`RESEARCH_BACKLOG_DRAINED`)
- [ ] `experiment list --status open` returns no entries with `priority ≤ 2`
- [ ] `experiment list --status active` is empty (nothing stuck claimed)
- [ ] Every completed experiment has: `MATH.md`, `run_experiment.py`, `results.json`, `PAPER.md`, `REVIEW-adversarial.md`, `LEARNINGS.md`
