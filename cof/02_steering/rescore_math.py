"""Re-score MATH records whose ``correct`` value is missing.

The original JSONL files are kept byte-for-byte intact.  A compact override file is
written next to them and is picked up automatically by the plotting scripts.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent / "vika" / "01_eval"))

from math_scoring import math_equal  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory holding steering*.jsonl")
    parser.add_argument("--steering-version", type=int, default=None, help="Default: newest version present")
    parser.add_argument("--out", type=Path, default=None, help="Default: RESULTS/math-correctness.json")
    return parser.parse_args()


def load_latest(results: Path, version: int | None) -> tuple[int, list[dict[str, Any]]]:
    everything: list[dict[str, Any]] = []
    for path in sorted(results.glob("steering*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    everything.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    versions = collections.Counter(row.get("steering_version") for row in everything)
    known = [value for value in versions if value is not None]
    if not known:
        raise SystemExit(f"No versioned records under {results}")
    wanted = max(known) if version is None else version
    deduplicated = {
        row["key"]: row
        for row in everything
        if row.get("steering_version") == wanted and row.get("benchmark") == "math_500"
    }
    if not deduplicated:
        raise SystemExit(f"No MATH-500 records for steering_version {wanted}")
    return wanted, list(deduplicated.values())


def main() -> None:
    args = parse_args()
    version, rows = load_latest(args.results, args.steering_version)
    scores: dict[str, dict[str, Any]] = {}
    attempted = 0
    for index, row in enumerate(rows, 1):
        # Unclosed generations have no final answer and were intentionally recorded as
        # false.  Only the missing labels reveal the absent math-verify dependency.
        if row.get("correct") is None:
            value = math_equal(row["output"], row["reference"])
            if value is None:
                raise SystemExit("math-verify is not installed; install the project dependencies first")
            scores[row["key"]] = {
                "correct": value,
                "output_sha256": hashlib.sha256(row["output"].encode("utf-8")).hexdigest(),
            }
            attempted += 1
        if index % 100 == 0 or index == len(rows):
            print(f"{index}/{len(rows)} records, {attempted} re-scored", flush=True)

    destination = args.out or args.results / "math-correctness.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "math_500",
        "steering_version": version,
        "records": len(rows),
        "rescored": len(scores),
        "correct": sum(item["correct"] for item in scores.values()),
        "scores": scores,
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{sum(item['correct'] for item in scores.values())}/{len(scores)} re-scored closed generations are correct")
    print(f"overrides -> {destination}")


if __name__ == "__main__":
    main()
