"""Streamlit dashboard for micro + macro arena results.

Usage:
    uv run --with streamlit python dashboard.py
"""

import json
import subprocess
import sys
from pathlib import Path


def load_macro_results() -> list[dict]:
    path = Path("macro_results.json")
    if path.exists():
        return json.loads(path.read_text())
    return []


def load_micro_tree() -> dict:
    path = Path("micro_model_tree.json")
    if path.exists():
        return json.loads(path.read_text())
    return {"models": {}}


def app():
    import streamlit as st

    st.set_page_config(page_title="LLM Arena", layout="wide")
    st.title("LLM Arena Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["Leaderboard", "Comparison", "Lineage", "Run History"])

    macro_results = load_macro_results()
    micro_tree = load_micro_tree()

    # --- Leaderboard ---
    with tab1:
        st.header("Leaderboard")

        # Macro results
        if macro_results:
            st.subheader("Macro (Pretrained Models)")

            # Best per model
            best = {}
            for r in macro_results:
                name = r["model_name"]
                ppl = r.get("perplexity", {}).get("python", float("inf"))
                if name not in best or ppl < best[name].get("perplexity", {}).get("python", float("inf")):
                    best[name] = r

            rows = []
            for r in sorted(best.values(), key=lambda x: x.get("perplexity", {}).get("python", float("inf"))):
                rows.append({
                    "Model": r["model_name"],
                    "Tier": r.get("tier", "?"),
                    "PPL (Python)": r.get("perplexity", {}).get("python"),
                    "Params": f"{r['param_count']:,}",
                    "tok/s": r.get("tokens_per_sec", 0),
                    "Load (s)": r.get("load_time_s", 0),
                    "Eval (s)": r.get("eval_time_s", 0),
                    "Memory (GB)": r.get("peak_memory_gb", 0),
                })
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No macro results yet. Run: `python run_macro.py --eval smollm-135m`")

        # Micro results
        micro_models = micro_tree.get("models", {})
        if micro_models:
            st.subheader("Micro (Char-Level Models)")
            micro_rows = []
            for name, info in micro_models.items():
                runs = info.get("runs", [])
                if runs:
                    best_run = min(runs, key=lambda r: r.get("val_loss") or r.get("final_loss", 999))
                    micro_rows.append({
                        "Model": name,
                        "Val Loss": best_run.get("val_loss"),
                        "Params": f"{best_run.get('param_count', 0):,}",
                        "tok/s": best_run.get("tokens_per_sec", 0),
                        "Time (s)": best_run.get("elapsed_s", 0),
                    })
            if micro_rows:
                st.dataframe(micro_rows, use_container_width=True)

    # --- Comparison Charts ---
    with tab2:
        st.header("Model Comparison")

        if macro_results:
            best = {}
            for r in macro_results:
                name = r["model_name"]
                ppl = r.get("perplexity", {}).get("python", float("inf"))
                if name not in best or ppl < best[name].get("perplexity", {}).get("python", float("inf")):
                    best[name] = r

            models = sorted(best.values(), key=lambda x: x.get("perplexity", {}).get("python", float("inf")))

            # Perplexity chart
            st.subheader("Python Perplexity (lower is better)")
            ppl_data = {r["model_name"]: r.get("perplexity", {}).get("python", 0) for r in models}
            st.bar_chart(ppl_data)

            # Speed chart
            st.subheader("Throughput (tok/s, higher is better)")
            speed_data = {r["model_name"]: r.get("tokens_per_sec", 0) for r in models}
            st.bar_chart(speed_data)

            # Param count chart
            st.subheader("Parameter Count")
            param_data = {r["model_name"]: r["param_count"] for r in models}
            st.bar_chart(param_data)

            # Efficiency: PPL / param_count
            st.subheader("Efficiency (PPL per Billion Params, lower is better)")
            eff_data = {}
            for r in models:
                ppl = r.get("perplexity", {}).get("python", 0)
                params_b = r["param_count"] / 1e9
                if params_b > 0 and ppl > 0:
                    eff_data[r["model_name"]] = ppl / params_b
            if eff_data:
                st.bar_chart(eff_data)
        else:
            st.info("No macro results yet.")

    # --- Lineage ---
    with tab3:
        st.header("Model Lineage")

        if micro_models:
            st.subheader("Micro Model Tree")
            # Build tree text
            children: dict[str | None, list[str]] = {}
            for name, info in micro_models.items():
                p = info.get("parent")
                children.setdefault(p, []).append(name)

            def render_tree(name: str, depth: int = 0) -> str:
                indent = "  " * depth
                runs = micro_models[name].get("runs", [])
                best = None
                if runs:
                    best = min(r.get("val_loss") or r.get("final_loss", 999) for r in runs)
                best_str = f" (best={best:.4f})" if best is not None else ""
                lines = [f"{indent}- **{name}**{best_str}"]
                for child in children.get(name, []):
                    lines.append(render_tree(child, depth + 1))
                return "\n".join(lines)

            roots = children.get(None, [])
            tree_text = "\n".join(render_tree(r) for r in roots)
            st.markdown(tree_text)
        else:
            st.info("No micro model tree yet. Run: `python run_arena.py --compare gpt,moe,moe-freeze --mode multidomain`")

        if macro_results:
            st.subheader("Macro Model Catalog")
            from macro.models import MODEL_CATALOG
            catalog_rows = []
            for name, info in MODEL_CATALOG.items():
                evaluated = any(r["model_name"] == name for r in macro_results)
                catalog_rows.append({
                    "Model": name,
                    "HF ID": info["hf_id"],
                    "Tier": info["tier"],
                    "Evaluated": "yes" if evaluated else "no",
                })
            st.dataframe(catalog_rows, use_container_width=True)

    # --- Run History ---
    with tab4:
        st.header("Run History")

        if macro_results:
            st.subheader("Macro Runs")
            history_rows = []
            for r in reversed(macro_results):
                history_rows.append({
                    "Timestamp": r.get("timestamp", "?"),
                    "Model": r["model_name"],
                    "PPL (Python)": r.get("perplexity", {}).get("python"),
                    "tok/s": r.get("tokens_per_sec", 0),
                    "Load (s)": r.get("load_time_s", 0),
                    "Eval (s)": r.get("eval_time_s", 0),
                })
            st.dataframe(history_rows, use_container_width=True)
        else:
            st.info("No runs recorded yet.")


if __name__ == "__main__":
    # When run directly, launch streamlit
    if "streamlit" not in sys.modules:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        app()
