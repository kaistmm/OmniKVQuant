#!/usr/bin/env python
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def select_clips(source):
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        rows = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            rows = list(rows.values())
    unique = {}
    for item in rows:
        if item.get("split") != "train":
            continue
        identity = str(item.get("id") or item.get("video") or "")
        if identity:
            unique.setdefault(identity, item)
    classes = defaultdict(list)
    for item in unique.values():
        label = str(item.get("class_label", "")).strip()
        if not label:
            raise ValueError("Every training row requires a class_label")
        classes[label].append(item)
    if not classes:
        raise ValueError("No VGGSound training clips found")
    rng = random.Random(42)
    selected = [rng.choice(classes[label]) for label in sorted(classes)]
    rng.shuffle(selected)
    result = []
    for item in selected:
        row = dict(
            id=str(item.get("id") or Path(item["video"]).stem),
            video=Path(item["video"]).name,
            split="train",
            class_label=item["class_label"],
        )
        if item.get("conversations"):
            row["conversations"] = item["conversations"]
        for field in ("caption", "answer", "reference"):
            if item.get(field):
                row[field] = item[field]
        result.append(row)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select one VGGSound training clip per class, using seed 42."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = select_clips(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Selected {len(rows)} classes -> {args.output}")
