"""CLI for the macro arena — pretrained coding model evaluation.

Usage:
    uv run --with mlx,mlx-lm,datasets,typer,python-dotenv python run_macro.py --help
    uv run --with mlx,mlx-lm,datasets,typer,python-dotenv python run_macro.py eval smollm-135m
    uv run --with mlx,mlx-lm,datasets,typer,python-dotenv python run_macro.py compare smollm-135m qwen-coder-0.5b
    micro-paper test moe
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")

import typer

app = typer.Typer(
    name="micro-paper",
    help="MLX Arena — micro/macro model training, evaluation, and testing.",
    no_args_is_help=True,
    add_completion=False,
)

MICRO_MODELS_DIR = Path(__file__).parent / "micro" / "models"

RESULTS_PATH = Path("macro_results.json")

# Shared option types
Langs = Annotated[str, typer.Option(help="Comma-separated languages")]
Samples = Annotated[int, typer.Option("--samples", help="Eval snippets per language")]
MaxTokens = Annotated[int, typer.Option("--max-tokens", help="Max tokens per snippet")]


def _load_results() -> list[dict]:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return []


def _save_result(result_dict: dict):
    results = _load_results()
    results.append(result_dict)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


def _print_leaderboard(results: list[dict]):
    if not results:
        print("No results yet. Run `eval` or `compare` first.")
        return

    best = {}
    for r in results:
        name = r["model_name"]
        ppl = r.get("perplexity", {}).get("python", float("inf"))
        if name not in best or ppl < best[name].get("perplexity", {}).get("python", float("inf")):
            best[name] = r

    ranked = sorted(best.values(), key=lambda r: r.get("perplexity", {}).get("python", float("inf")))

    header = f"{'Rank':>4} | {'Model':<20} | {'Tier':<8} | {'PPL(py)':>8} | {'Params':>12} | {'tok/s':>8} | {'Load':>6} | {'Eval':>6} | {'Mem':>7} | {'Via':<5}"
    print(f"\n{'='*len(header)}")
    print("MACRO LEADERBOARD")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for i, r in enumerate(ranked, 1):
        ppl_py = r.get("perplexity", {}).get("python", float("inf"))
        ppl_str = f"{ppl_py:.2f}" if ppl_py < float("inf") else "N/A"
        backend = r.get("backend", "local")
        via = "API" if backend == "api" else "MLX"
        arch = r.get("arch", "")
        if arch:
            via = f"{via}/{arch}"
        print(
            f"{i:>4} | {r['model_name']:<20} | {r.get('tier', '?'):<8} | "
            f"{ppl_str:>8} | {r['param_count']:>12,} | "
            f"{r.get('tokens_per_sec', 0):>8.0f} | "
            f"{r.get('load_time_s', 0):>5.1f}s | "
            f"{r.get('eval_time_s', 0):>5.1f}s | "
            f"{r.get('peak_memory_gb', 0):>5.2f}G | {via}"
        )

    all_langs = set()
    for r in ranked:
        all_langs.update(r.get("perplexity", {}).keys())
    if len(all_langs) > 1:
        print(f"\nPer-language perplexity:")
        for r in ranked:
            ppls = r.get("perplexity", {})
            parts = [f"{lang}={ppl:.2f}" for lang, ppl in sorted(ppls.items())]
            print(f"  {r['model_name']}: {', '.join(parts)}")


@app.command()
def test(
    model: Annotated[Optional[str], typer.Argument(help="Model name (gpt, moe, moe_freeze). Omit to run all.")] = None,
):
    """Run test suite for a micro model (or all models)."""
    if model:
        test_file = MICRO_MODELS_DIR / model / f"test_{model}.py"
        if not test_file.exists():
            print(f"No test file found at {test_file}")
            raise typer.Exit(1)
        targets = [test_file]
    else:
        targets = sorted(MICRO_MODELS_DIR.glob("*/test_*.py"))
        if not targets:
            print("No test files found.")
            raise typer.Exit(1)

    failed = []
    for tf in targets:
        name = tf.parent.name
        print(f"\n{'='*60}")
        print(f"  Running tests: {name}")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, str(tf)])
        if result.returncode != 0:
            failed.append(name)

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        raise typer.Exit(1)
    else:
        print(f"All {len(targets)} test suite(s) passed.")


@app.command()
def ls():
    """List all models in the catalog."""
    from macro.models import MODEL_CATALOG

    api_ok = bool(os.environ.get("TOGETHER_API_KEY"))

    print(f"  {'Name':<20} {'ID':<55} {'Tier':<8} {'Backend'}")
    print(f"  {'-'*20} {'-'*55} {'-'*8} {'-'*10}")
    for name, info in MODEL_CATALOG.items():
        model_id = info.get("hf_id") or info.get("together_id", "?")
        backend = info.get("backend", "local")
        arch = f" ({info['arch']})" if info.get("arch") else ""
        marker = ""
        if backend == "api" and not api_ok:
            marker = " [no key]"
        print(f"  {name:<20} {model_id:<55} {info['tier']:<8} {backend}{arch}{marker}")

    if not api_ok:
        print(f"\n  Tip: add TOGETHER_API_KEY to .env to enable API models.")


@app.command()
def eval(
    model: Annotated[str, typer.Argument(help="Model name from catalog")],
    langs: Langs = "python",
    samples: Samples = 100,
    max_tokens: MaxTokens = 512,
):
    """Evaluate a single model from the catalog."""
    from macro.eval import eval_model

    lang_list = [l.strip() for l in langs.split(",")]
    result = eval_model(model, langs=lang_list, n_samples=samples, max_tokens=max_tokens)
    _save_result(result.to_dict())
    print(f"\nSaved to {RESULTS_PATH}")
    _print_leaderboard(_load_results())


@app.command()
def eval_hf(
    hf_id: Annotated[str, typer.Argument(help="HuggingFace model ID (e.g. mlx-community/SomeModel-4bit)")],
    name: Annotated[Optional[str], typer.Option(help="Short name for this model")] = None,
    langs: Langs = "python",
    samples: Samples = 100,
    max_tokens: MaxTokens = 512,
):
    """Evaluate an arbitrary HuggingFace model (not in catalog)."""
    from macro.eval import eval_model

    display_name = name or hf_id.split("/")[-1]
    lang_list = [l.strip() for l in langs.split(",")]
    result = eval_model(display_name, hf_id=hf_id, langs=lang_list, n_samples=samples, max_tokens=max_tokens)
    _save_result(result.to_dict())
    print(f"\nSaved to {RESULTS_PATH}")
    _print_leaderboard(_load_results())


@app.command()
def compare(
    models: Annotated[list[str], typer.Argument(help="Model names to compare")],
    langs: Langs = "python",
    samples: Samples = 100,
    max_tokens: MaxTokens = 512,
):
    """Compare multiple models head-to-head."""
    from macro.eval import eval_model

    lang_list = [l.strip() for l in langs.split(",")]
    results = []
    for m in models:
        result = eval_model(m, langs=lang_list, n_samples=samples, max_tokens=max_tokens)
        _save_result(result.to_dict())
        results.append(result)
    print(f"\nAll saved to {RESULTS_PATH}")
    _print_leaderboard([r.to_dict() for r in results])


@app.command()
def leaderboard():
    """Show the leaderboard from all stored results."""
    _print_leaderboard(_load_results())


if __name__ == "__main__":
    app()
