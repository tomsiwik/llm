"""
exp_hedgehog_domain_adapter_js — SCAFFOLD

Distill JavaScript domain nuance from MDN+Eloquent-JS teacher into rank-8 LoRA.
Structurally identical to exp_hedgehog_procedural_adapter_refactor (same
loss, same capture pattern) — see that experiment's run_experiment.py for the
shared training machinery. Only the dataset changes.

Grounded: arxiv:2604.14191 §3.1; arxiv:2402.04347.

!!!  invoke /mlx-dev and /fast-mlx before coding (ralph guardrail).  !!!
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
TEACHER_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
ADAPTER_RANK = 8
ADAPTER_TARGETS = ("v_proj", "o_proj")
SEQLEN = 2048  # docs blocks are longer than refactor snippets

JS_FOCUS_TOPICS = [
    "hoisting_and_tdz",
    "closures_and_scope",
    "this_binding_and_arrow_functions",
    "event_loop_microtasks",
    "prototype_chain",
    "async_await_error_handling",
]


def load_js_corpus() -> list[dict]:
    """Scrape + generate Q-A pairs for each focus topic.

    Returns [{topic, source_text, question, reference_answer}, ...].

    TODO(researcher):
      - Scrape MDN pages for each topic (respect robots.txt)
      - Download Eloquent-JS chapters (CC-BY-NC 3.0) from eloquentjavascript.net
      - Use larger model with source in context to generate ~30 Q-A per topic
      - Dedup + 80/20 split
    """
    raise NotImplementedError("data pipeline — researcher hat")


def main():
    """See exp_hedgehog_procedural_adapter_refactor.run_experiment.main() —
    structure is identical, swap load_fowler_examples → load_js_corpus,
    swap eval_k2_refactor_quality → eval_k2_js_bench,
    swap eval_k4_nonrefactor_codegen → eval_k4_mmlu_drop,
    swap eval_k3_humaneval → eval_k3_humaneval (Python, unchanged — JS adapter
    should NOT help Python).
    """
    raise NotImplementedError("main — researcher hat (pattern from refactor exp)")


if __name__ == "__main__":
    main()
