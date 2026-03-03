#!/usr/bin/env python3
"""Benchmark status probe — check progress from log + results JSON.

Usage:
    python torch_bench/status.py                    # on the pod
    ssh <pod> python /workspace/llm/torch_bench/status.py  # remote
"""

import json
import os
import re
import sys
import time

WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
LOG = os.path.join(WORKSPACE, "bench.log")
RESULTS = os.path.join(WORKSPACE, "results_7b.json")
PID_FILE = os.path.join(WORKSPACE, "bench.pid")

ALL_CONFIGS = ["lora_seqft", "lora_ewc", "lora_replay", "lora_olora", "lora_lifecycle"]
DOMAINS = ["python", "javascript", "rust", "sql", "cpp"]


def check_alive():
    """Check if benchmark process is still running."""
    if not os.path.exists(PID_FILE):
        return None
    with open(PID_FILE) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def parse_log():
    """Parse bench.log for current progress."""
    if not os.path.exists(LOG):
        return {"status": "no log file"}

    with open(LOG) as f:
        lines = f.readlines()

    if not lines:
        return {"status": "empty log"}

    info = {
        "current_config": None,
        "current_task": None,
        "current_domain": None,
        "last_step": None,
        "last_loss": None,
        "last_ppl": None,
        "completed_configs": [],
        "evaluating": False,
        "errors": [],
    }

    for line in lines:
        line = line.strip()

        # Config header
        m = re.search(r"CONFIG:\s+(\w+)", line)
        if m:
            info["current_config"] = m.group(1)

        # Task/domain
        m = re.search(r"Task (\d+): Train on '(\w+)'", line)
        if m:
            info["current_task"] = int(m.group(1))
            info["current_domain"] = m.group(2)

        # Training step
        m = re.search(r"step\s+(\d+)/(\d+):\s+loss=([\d.]+),\s+ppl=([\d.inf]+)", line)
        if m:
            info["last_step"] = f"{m.group(1)}/{m.group(2)}"
            info["last_loss"] = m.group(3)
            info["last_ppl"] = m.group(4)

        # Evaluating
        if "Evaluating" in line:
            info["evaluating"] = True
        elif "step" in line:
            info["evaluating"] = False

        # Checkpoint saved
        m = re.search(r"Checkpoint saved \((\w+) done\)", line)
        if m:
            info["completed_configs"].append(m.group(1))

        # Errors
        if "Error" in line or "Traceback" in line or "CUDA" in line:
            info["errors"].append(line)

    return info


def load_results():
    """Load completed results from JSON."""
    if not os.path.exists(RESULTS):
        return None
    try:
        with open(RESULTS) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def format_forgetting(results_data):
    """Format forgetting summary from results."""
    lines = []
    for res in results_data.get("results", []):
        fgt = res.get("forgetting", {})
        if fgt:
            pcts = [v["percent"] for v in fgt.values()]
            mean_fgt = sum(pcts) / len(pcts) if pcts else 0
            lines.append(f"    {res['name']:>20s}: mean forgetting = {mean_fgt:+.1f}%"
                         f"  ({res.get('elapsed_s', 0):.0f}s)")
    return lines


def main():
    pid = check_alive()
    log_info = parse_log()
    results = load_results()

    print("=" * 60)
    print("  7B CL Benchmark Status")
    print("=" * 60)

    # Process status
    if pid:
        print(f"  Process: RUNNING (PID {pid})")
    elif os.path.exists(PID_FILE):
        print(f"  Process: STOPPED (was running)")
    else:
        print(f"  Process: NOT STARTED")

    # Log file
    if os.path.exists(LOG):
        size = os.path.getsize(LOG)
        mtime = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(LOG)))
        print(f"  Log: {size:,} bytes, last modified {mtime}")
    else:
        print(f"  Log: not found")

    # Completed configs
    completed = set()
    if results:
        completed = {r["name"] for r in results.get("results", [])}
    print(f"\n  Completed: {len(completed)}/{len(ALL_CONFIGS)}")
    for c in ALL_CONFIGS:
        status = "done" if c in completed else "pending"
        marker = "[x]" if c in completed else "[ ]"
        print(f"    {marker} {c}")

    # Current activity
    if log_info.get("current_config"):
        cfg = log_info["current_config"]
        if cfg not in completed:
            task = log_info.get("current_task", "?")
            domain = log_info.get("current_domain", "?")
            step = log_info.get("last_step", "?")
            loss = log_info.get("last_loss", "?")
            ppl = log_info.get("last_ppl", "?")

            print(f"\n  Currently: {cfg}")
            print(f"    Task {task}/{len(DOMAINS)-1}: {domain}")
            if log_info.get("evaluating"):
                print(f"    Status: evaluating...")
            else:
                print(f"    Step: {step}  loss={loss}  ppl={ppl}")

    # Results summary
    if results:
        fgt_lines = format_forgetting(results)
        if fgt_lines:
            print(f"\n  Results so far:")
            for line in fgt_lines:
                print(line)

    # Errors
    errors = log_info.get("errors", [])
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[-5:]:  # last 5
            print(f"    {e[:100]}")

    # ETA estimate
    if results and completed and pid:
        elapsed_per_config = []
        for res in results.get("results", []):
            if res.get("elapsed_s"):
                elapsed_per_config.append(res["elapsed_s"])
        if elapsed_per_config:
            avg = sum(elapsed_per_config) / len(elapsed_per_config)
            remaining_count = len(ALL_CONFIGS) - len(completed)
            # Subtract 1 if current config is in progress
            if log_info.get("current_config") and log_info["current_config"] not in completed:
                remaining_count -= 0.5  # rough halfway estimate
            eta_s = avg * remaining_count
            eta_h = eta_s / 3600
            print(f"\n  ETA: ~{eta_h:.1f} hours ({avg/3600:.1f}h per config avg)")

    print("=" * 60)


if __name__ == "__main__":
    main()
