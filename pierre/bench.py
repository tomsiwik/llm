"""Pierre Bench — standardized experiment output and benchmarks.

Every experiment imports this. It handles:
  - Structured results collection
  - Standard benchmarks via lm-evaluation-harness (MMLU, GSM8K, etc.)
  - Custom benchmarks (PPL, factual recall, code syntax, behavioral)
  - Speed measurement
  - Memory tracking
  - Kill criteria evaluation
  - Consistent JSON output + human-readable summary

Standard benchmarks use lm-evaluation-harness (EleutherAI) via mlx_lm.evaluate.
This is the SAME evaluation pipeline used by the Open LLM Leaderboard.
Do NOT hand-roll benchmark implementations — use lm_eval() for reproducibility.

Usage:
    from pierre.bench import Experiment

    exp = Experiment("my_experiment")

    with exp.phase("baseline"):
        ppl = exp.ppl(model, tok, texts)
        exp.metric("base_ppl", ppl)

    with exp.phase("standard_benchmarks"):
        scores = exp.lm_eval("mlx-community/Qwen3-4B-4bit",
                             tasks=["mmlu", "gsm8k_cot", "hellaswag"],
                             limit=100)

    with exp.phase("speed"):
        tps = exp.speed(model, tok, prompt)

    exp.kill_if("base_ppl", ">", 10.0, kid="K801")
    exp.save()
"""

import ast
import gc
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ── Stop words for factual recall ────────────────────────────────────────

_STOP = frozenset({
    'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might','can',
    'to','of','in','for','on','with','at','by','from','as','and','but','or','not',
    'so','yet','both','either','each','every','all','any','few','more','most','other',
    'some','such','no','only','own','same','than','too','very','just','because','if',
    'when','where','how','what','which','who','this','that','these','those','it','its',
    'i','me','my','we','our','you','your','he','him','his','she','her','they','them','their',
})


# ── JSON encoder ─────────────────────────────────────────────────────────

class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, mx.array): return o.tolist()
        return super().default(o)


# ── Standard metrics ─────────────────────────────────────────────────────

def ppl(model, tokenizer, texts, max_seq=256):
    """Compute mean perplexity over a list of texts."""
    total_loss, total_tokens = 0.0, 0
    for text in texts:
        toks = tokenizer.encode(text)[:max_seq]
        if len(toks) < 4:
            continue
        x = mx.array(toks)[None, :]
        logits = model(x)
        mx.eval(logits)
        targets = x[:, 1:]
        lp = mx.log(mx.softmax(logits[:, :-1, :], axis=-1) + 1e-10)
        tlp = mx.take_along_axis(lp, targets[:, :, None], axis=-1).squeeze(-1)
        mx.eval(tlp)
        total_loss += -tlp.sum().item()
        total_tokens += targets.shape[1]
        del logits, lp, tlp, x
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


def factual_recall(generated, reference):
    """Token-overlap factual recall (stop words removed, min length 3)."""
    def toks(text):
        return set(w for w in re.findall(r'\b[a-z]+\b', text.lower())
                   if w not in _STOP and len(w) > 2)
    g, r = toks(generated), toks(reference)
    return len(g & r) / len(r) if r else 0.0


def code_syntax(text):
    """Check if generated text contains valid Python syntax."""
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
    code = '\n'.join(blocks) if blocks else '\n'.join(
        l for l in text.split('\n') if l.strip() and not l.startswith('#'))
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def behavioral_score(generated, reference, domain):
    """Domain-appropriate behavioral score in [0, 1]."""
    if domain == "code":
        return 0.7 * float(code_syntax(generated)) + 0.3 * factual_recall(generated, reference)
    return factual_recall(generated, reference)


def mmlu_eval(model, tokenizer, questions, max_seq=512):
    """Evaluate MMLU-style questions. Returns (correct, total, per_subject).

    questions: list of (subject, question, choices, answer) tuples
    """
    correct, total = 0, 0
    per_subject = {}

    for item in questions:
        if len(item) == 4:
            subject, question, choices, answer = item
        elif len(item) == 3:
            question, choices, answer = item
            subject = "general"
        else:
            continue

        prompt = f"Question: {question}\n{choices}\nAnswer: The correct answer is"
        tokens = tokenizer.encode(prompt)[:max_seq]
        logits = model(mx.array(tokens)[None, :])
        mx.eval(logits)
        last = logits[0, -1]

        preds = {}
        for letter in ["A", "B", "C", "D"]:
            t = tokenizer.encode(f" {letter}")
            preds[letter] = last[t[0]].item() if t else -999

        predicted = max(preds, key=preds.get)
        is_correct = predicted == answer
        if is_correct:
            correct += 1
        total += 1

        if subject not in per_subject:
            per_subject[subject] = {"correct": 0, "total": 0}
        per_subject[subject]["total"] += 1
        if is_correct:
            per_subject[subject]["correct"] += 1

        del logits

    for s in per_subject:
        per_subject[s]["accuracy"] = round(
            per_subject[s]["correct"] / per_subject[s]["total"], 3
        ) if per_subject[s]["total"] > 0 else 0.0

    return correct, total, per_subject


def speed(model, tokenizer, prompt, max_tokens=128, n_warmup=3, n_measure=5, temp=0.0):
    """Measure generation speed in tok/s.

    Returns dict with tok_per_s, runs, peak_memory_gb.
    """
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=temp)

    for _ in range(n_warmup):
        mlx_generate(model, tokenizer, prompt=prompt, max_tokens=32,
                     sampler=sampler, verbose=False)

    mx.reset_peak_memory()
    runs = []
    for _ in range(n_measure):
        t0 = time.time()
        out = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                           sampler=sampler, verbose=False)
        dt = time.time() - t0
        n = len(tokenizer.encode(out)) - len(tokenizer.encode(prompt))
        runs.append({"time_s": round(dt, 3), "tokens": n})

    total_toks = sum(r["tokens"] for r in runs)
    total_time = sum(r["time_s"] for r in runs)
    tps = total_toks / total_time if total_time > 0 else 0
    peak = mx.get_peak_memory() / 1e9

    return {
        "tok_per_s": round(tps, 1),
        "peak_memory_gb": round(peak, 2),
        "runs": runs,
    }


# ── Experiment class ─────────────────────────────────────────────────────

class Experiment:
    """Structured experiment runner with automatic metric collection.

    Usage:
        exp = Experiment("my_experiment", dir="micro/models/my_exp/")

        with exp.phase("baseline"):
            val = some_measurement()
            exp.metric("baseline_ppl", val)

        exp.kill_if("baseline_ppl", ">", 10.0, kid="K801")
        exp.save()
    """

    def __init__(self, name, dir=None):
        self.name = name
        self.dir = Path(dir) if dir else Path(f"micro/models/{name}/")
        self.results = {
            "experiment": name,
            "metrics": {},
            "phases": {},
            "kill_criteria": {},
        }
        self._start_time = time.time()
        self._current_phase = None
        self._phase_start = None
        print(f"\n{'='*60}", flush=True)
        print(f"Experiment: {name}", flush=True)
        print(f"{'='*60}", flush=True)

    # ── Phase context manager ────────────────────────────────

    class _Phase:
        def __init__(self, exp, name):
            self.exp = exp
            self.name = name

        def __enter__(self):
            self.exp._current_phase = self.name
            self.exp._phase_start = time.time()
            self.exp.results["phases"][self.name] = {"start": time.time()}
            print(f"\n=== {self.name} ===", flush=True)
            return self

        def __exit__(self, *exc):
            dt = time.time() - self.exp._phase_start
            self.exp.results["phases"][self.name]["duration_s"] = round(dt, 2)
            self.exp.results["phases"][self.name]["memory_gb"] = round(
                mx.get_active_memory() / 1e9, 2)
            print(f"  [{self.name}] done in {dt:.1f}s", flush=True)
            self.exp._current_phase = None

    def phase(self, name):
        return self._Phase(self, name)

    # ── Metric recording ─────────────────────────────────────

    def metric(self, name, value, display=True):
        """Record a metric. Auto-rounds floats."""
        if isinstance(value, float):
            value = round(value, 4)
        self.results["metrics"][name] = value
        if display:
            print(f"  {name}: {value}", flush=True)
        return value

    def metrics(self, **kwargs):
        """Record multiple metrics at once."""
        for k, v in kwargs.items():
            self.metric(k, v)

    # ── Standard benchmarks (convenience wrappers) ───────────

    def ppl(self, model, tokenizer, texts, name=None, max_seq=256):
        """Compute and record PPL."""
        val = ppl(model, tokenizer, texts, max_seq)
        if name:
            self.metric(name, val)
        return val

    def speed(self, model, tokenizer, prompt="Explain machine learning.", name=None, **kwargs):
        """Measure and record speed."""
        result = speed(model, tokenizer, prompt, **kwargs)
        if name:
            self.metric(f"{name}_tok_per_s", result["tok_per_s"])
            self.metric(f"{name}_peak_gb", result["peak_memory_gb"])
        return result

    def mmlu(self, model, tokenizer, questions, name=None, **kwargs):
        """Run MMLU and record results."""
        c, t, per_sub = mmlu_eval(model, tokenizer, questions, **kwargs)
        acc = c / t if t > 0 else 0.0
        if name:
            self.metric(f"{name}_accuracy", acc)
            self.metric(f"{name}_correct", c)
            self.metric(f"{name}_total", t)
        return acc, c, t, per_sub

    def lm_eval(self, model_id, tasks, limit=None, batch_size=4, num_shots=None, name=None):
        """Run standard benchmarks via lm-evaluation-harness.

        Uses the SAME evaluation pipeline as the Open LLM Leaderboard.
        Results are directly comparable to published numbers.

        Args:
            model_id: HuggingFace model ID or local path (must be MLX-compatible)
            tasks: list of task names (e.g., ["mmlu", "gsm8k_cot", "hellaswag"])
            limit: max examples per task (None = full dataset)
            batch_size: evaluation batch size
            num_shots: few-shot count (None = task default)
            name: prefix for recorded metrics

        Returns:
            dict of {task_name: {metric: value}} from lm-eval
        """
        from mlx_lm.evaluate import MLXLM
        from lm_eval import simple_evaluate

        print(f"  lm_eval: {tasks} on {model_id} (limit={limit})", flush=True)
        model = MLXLM(model_id)

        kwargs = {
            "model": model,
            "tasks": tasks,
            "batch_size": batch_size,
        }
        if limit is not None:
            kwargs["limit"] = limit
        if num_shots is not None:
            kwargs["num_fewshot"] = num_shots

        raw = simple_evaluate(**kwargs)
        results = {}
        for task, metrics in raw.get("results", {}).items():
            results[task] = {}
            for k, v in metrics.items():
                if k.startswith("_") or k == "alias":
                    continue
                results[task][k] = v
                metric_name = f"{name}_{task}_{k}" if name else f"{task}_{k}"
                # Only record the main metric (not stderr)
                if "stderr" not in k:
                    self.metric(metric_name, v)

        return results

    def behavioral(self, model, tokenizer, texts, domain, name=None, max_tokens=128):
        """Generate and evaluate behavioral quality."""
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=0.0)

        scores = []
        for text in texts:
            if "### Response:" in text:
                prompt = text.split("### Response:")[0].strip() + "\n### Response:\n"
                ref = text.split("### Response:")[-1].strip()
            else:
                prompt, ref = text[:200], text
            try:
                gen = mlx_generate(model, tokenizer, prompt=prompt,
                                   max_tokens=max_tokens, sampler=sampler, verbose=False)
                scores.append(behavioral_score(gen, ref, domain))
            except Exception:
                scores.append(0.0)

        mean = float(np.mean(scores)) if scores else 0.0
        if name:
            self.metric(f"{name}_behavioral", mean)
        return mean

    # ── Kill criteria ────────────────────────────────────────

    def kill_if(self, metric_name, op, threshold, kid=""):
        """Evaluate a kill criterion.

        op: "<", ">", "<=", ">="
        Returns True if criterion PASSES (not killed).
        """
        value = self.results["metrics"].get(metric_name)
        if value is None:
            self.results["kill_criteria"][kid] = {
                "pass": False, "metric": metric_name,
                "detail": f"metric '{metric_name}' not found"
            }
            return False

        ops = {"<": lambda v, t: v < t, ">": lambda v, t: v > t,
               "<=": lambda v, t: v <= t, ">=": lambda v, t: v >= t}
        # Kill if the condition is TRUE — so PASS means condition is FALSE
        killed = ops[op](value, threshold)
        passed = not killed

        self.results["kill_criteria"][kid] = {
            "pass": passed,
            "metric": metric_name,
            "value": value,
            "op": op,
            "threshold": threshold,
        }

        status = "PASS" if passed else "FAIL"
        print(f"  {kid}: {status} — {metric_name}={value} {op} {threshold}", flush=True)
        return passed

    # ── Save results ─────────────────────────────────────────

    def save(self):
        """Save results.json and print summary."""
        self.results["total_time_s"] = round(time.time() - self._start_time, 1)
        self.results["all_pass"] = all(
            v.get("pass", False) for v in self.results["kill_criteria"].values()
        )

        # Write JSON
        results_path = self.dir / "results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(self.results, indent=2, cls=_Enc))

        # Print summary
        print(f"\n{'='*60}", flush=True)
        print(f"Kill criteria:", flush=True)
        for k, v in self.results["kill_criteria"].items():
            s = "PASS" if v.get("pass") else "FAIL"
            print(f"  {k}: {s} — {v}", flush=True)

        status = "ALL PASS" if self.results["all_pass"] else "KILLED"
        print(f"\n{status} in {self.results['total_time_s']}s", flush=True)
        print(f"Results: {results_path}", flush=True)
        return self.results


# ── Utility: cleanup ─────────────────────────────────────────────────────

def cleanup(*objects):
    """Release MLX memory between phases."""
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
