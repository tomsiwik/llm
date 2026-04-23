#!/usr/bin/env python3
"""
Doom-loop detector for the ralph loop.
Adapted from huggingface/ml-intern agent/core/doom_loop.py (MIT).

Detects when the loop is stuck calling the same hat/event pattern and
emits a corrective prompt on stdout. Exit 0 = clean, exit 1 = stuck.

Usage (call at the start of each hat activation):
    python .ralph/tools/doom_loop.py            # checks the most recent events file
    python .ralph/tools/doom_loop.py --lookback 30
    python .ralph/tools/doom_loop.py --events-file .ralph/events-YYYYMMDD-HHMMSS.jsonl

The hat should read stdout; if non-empty, inject it into the assistant's
context as a system reminder before continuing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EventSignature:
    """Hashable signature for one loop event (hat + topic + payload-prefix hash)."""

    hat: str
    topic: str
    payload_hash: str


def _hash_payload(payload: str, prefix_chars: int = 512) -> str:
    """Short hash of the first prefix_chars of the payload.

    Prefix-only so minor timestamp/iteration drift in payloads doesn't defeat
    the detector for genuinely identical work.
    """
    return hashlib.md5(payload[:prefix_chars].encode("utf-8", errors="replace")).hexdigest()[:12]


def load_events(events_file: Path, lookback: int = 30) -> list[EventSignature]:
    """Read the last `lookback` events from a .ralph/events-*.jsonl file.

    Each line is one event: {ts, iteration, hat, topic, triggered, payload}.
    Malformed lines are skipped silently.
    """
    if not events_file.exists():
        return []

    sigs: list[EventSignature] = []
    with events_file.open() as f:
        lines = f.readlines()

    for raw in lines[-lookback:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        sigs.append(
            EventSignature(
                hat=str(e.get("hat", "")),
                topic=str(e.get("topic", "")),
                payload_hash=_hash_payload(str(e.get("payload", ""))),
            )
        )
    return sigs


def detect_identical_consecutive(
    sigs: list[EventSignature], threshold: int = 3
) -> EventSignature | None:
    """Return the signature if `threshold+` identical consecutive events occur."""
    if len(sigs) < threshold:
        return None
    count = 1
    for i in range(1, len(sigs)):
        if sigs[i] == sigs[i - 1]:
            count += 1
            if count >= threshold:
                return sigs[i]
        else:
            count = 1
    return None


def detect_repeating_sequence(sigs: list[EventSignature]) -> list[EventSignature] | None:
    """Detect repeating patterns (e.g. researcher.done → reviewer.revise → researcher.done
    → reviewer.revise) for sequences of length 2..5 with 2+ consecutive repetitions."""
    n = len(sigs)
    for seq_len in range(2, 6):
        min_required = seq_len * 2
        if n < min_required:
            continue
        tail = sigs[-min_required:]
        pattern = tail[:seq_len]
        reps = 0
        for start in range(n - seq_len, -1, -seq_len):
            chunk = sigs[start : start + seq_len]
            if chunk == pattern:
                reps += 1
            else:
                break
        if reps >= 2:
            return pattern
    return None


def _latest_events_file() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    candidates = sorted(root.glob("events-*.jsonl"))
    return candidates[-1] if candidates else None


def check(events_file: Path, lookback: int = 30, threshold: int = 3) -> str | None:
    sigs = load_events(events_file, lookback=lookback)
    if len(sigs) < threshold:
        return None

    hit = detect_identical_consecutive(sigs, threshold=threshold)
    if hit:
        return (
            f"[SYSTEM: DOOM LOOP DETECTED] The loop has emitted '{hit.hat}/{hit.topic}' "
            f"with the same payload at least {threshold} times in a row. "
            f"STOP repeating this approach — it is not producing new state on disk. "
            f"Diagnose what has not changed (did the experiment actually run? did the "
            f"CLI accept the status? is a dep still blocking?). Try a structurally "
            f"different action: claim a different experiment, release the current "
            f"claim, promote priority of a blocking dep, or emit `review.killed` "
            f"if the current target is unfixable. Text-only responses without a tool "
            f"call will end the loop permanently — always take an action."
        )

    pat = detect_repeating_sequence(sigs)
    if pat:
        desc = " → ".join(f"{s.hat}/{s.topic}" for s in pat)
        return (
            f"[SYSTEM: DOOM LOOP DETECTED] The loop is cycling through the pattern "
            f"[{desc}] without making progress on disk. STOP this cycle. Common "
            f"causes: (a) reviewer keeps issuing REVISE on the same KC the researcher "
            f"cannot satisfy at this scale — escalate to PROVISIONAL with a follow-up, "
            f"(b) dep-chain deadlock — check that parent experiments can actually "
            f"reach supported, (c) researcher running the same broken code after each "
            f"revise — read the REVIEW-adversarial.md top-3-fixes and apply them in "
            f"the file, not in a retry. On round 3, proceed with caveats per "
            f"reviewer.md §REVISE discipline."
        )
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect ralph-loop doom loops.")
    ap.add_argument("--events-file", type=Path, default=None,
                    help="Path to .ralph/events-*.jsonl (default: most recent).")
    ap.add_argument("--lookback", type=int, default=30,
                    help="How many trailing events to consider (default 30).")
    ap.add_argument("--threshold", type=int, default=3,
                    help="Identical-consecutive count that trips detection (default 3).")
    args = ap.parse_args()

    events_file = args.events_file or _latest_events_file()
    if events_file is None:
        print("", end="")
        return 0

    msg = check(events_file, lookback=args.lookback, threshold=args.threshold)
    if msg:
        print(msg)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
