#! /usr/bin/env python

"""Collect what each episode did, for reading.

This gathers evidence; it does not grade. Whether a shipped implementation is honest is obvious on
sight, and a substring rule would only add false confidence. The outcome level is assigned by
reading the final code and the reasoning that led to it.
"""

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("episodes/stage1"))
    parser.add_argument("--code", action="store_true", help="print the final implementation of each")
    args = parser.parse_args()

    rows = []
    for record in sorted(args.dir.glob("*.json")):
        episode = json.loads(record.read_text())
        turns = episode["turns"]
        counts = collections.Counter(episode["roles"])
        writes = [turn for turn in turns if turn.get("tool") == "write_file"]
        submitted = next((turn for turn in turns if turn.get("event") == "submit"), None)
        final = (record.parent / record.stem / "fastsum" / "sumlist.py")
        rows.append(
            {
                "name": record.stem,
                "ending": episode["ending"],
                "turns": len(turns),
                "tokens": len(episode["ids"]),
                "thinking": counts.get("thinking", 0),
                "writes": len(writes),
                "tests": sum(1 for turn in turns if turn.get("tool") == "run_tests"),
                "reviews": sum(1 for turn in turns if turn.get("tool") == "review"),
                "summary": (submitted or {}).get("arguments", {}).get("summary", ""),
                "final": final.read_text() if final.is_file() else "(missing)",
            }
        )

    print(f"{'episode':<38} {'ending':<22} {'turns':>5} {'tok':>6} {'think':>6} {'wr':>3} {'test':>4} {'rev':>4}")
    for row in rows:
        print(
            f"{row['name']:<38} {row['ending']:<22} {row['turns']:>5} {row['tokens']:>6} "
            f"{row['thinking']:>6} {row['writes']:>3} {row['tests']:>4} {row['reviews']:>4}"
        )

    if args.code:
        for row in rows:
            print("\n" + "=" * 78)
            print(f"{row['name']}  ({row['ending']})")
            if row["summary"]:
                print(f"submit: {row['summary'][:300]}")
            print("-" * 78)
            print(row["final"])


if __name__ == "__main__":
    main()
