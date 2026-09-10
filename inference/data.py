import json
import re
from pathlib import Path


def parse_choice(text):
    match = re.search("\\b([A-D])\\b", str(text).strip())
    return match.group(1) if match else None


def load_items(path):
    with open(path, encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list) or not items:
        raise ValueError("The manifest must be a nonempty JSON list")
    seen = set()
    for index, item in enumerate(items):
        item_id = str(item.get("id", index))
        if item_id in seen:
            raise ValueError(f"Duplicate sample id: {item_id}")
        seen.add(item_id)
        item["id"] = item_id
        turns = {turn["from"]: turn["value"] for turn in item["conversations"]}
        item["prompt"] = turns["human"]
        item["answer"] = parse_choice(turns["gpt"])
        if item["answer"] is None or not item.get("video"):
            raise ValueError(f"Invalid WorldSense sample: {item_id}")
    return items


def video_path(item, video_dir):
    path = Path(item["video"])
    if not path.is_absolute():
        path = Path(video_dir) / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
