#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.data import load_items


def score(path, data):
    gold = {item["id"]: item["answer"] for item in load_items(data)}
    seen = set()
    correct = invalid = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row["id"])
            if key in seen or key not in gold:
                raise ValueError(f"Duplicate or unknown sample id: {key}")
            if row["answer"] != gold[key]:
                raise ValueError(f"Ground truth mismatch: {key}")
            seen.add(key)
            pred = row.get("prediction")
            invalid += int(pred not in ("A", "B", "C", "D"))
            correct += int(pred == gold[key])
    if not seen:
        raise ValueError("No predictions to score")
    return dict(
        accuracy=100 * correct / len(seen),
        correct=correct,
        evaluated=len(seen),
        expected=len(gold),
        missing=len(gold) - len(seen),
        invalid=invalid,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score WorldSense JSONL; invalid predictions count as incorrect."
    )
    parser.add_argument("predictions")
    parser.add_argument(
        "--data", default=Path(__file__).resolve().parents[1] / "json/worldsense.json"
    )
    args = parser.parse_args()
    print(json.dumps(score(args.predictions, args.data), indent=2))
