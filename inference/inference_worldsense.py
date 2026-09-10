#!/usr/bin/env python
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.data import load_items, parse_choice, video_path


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-Omni on WorldSense with a packed 2-bit KV cache."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument(
        "--method", choices=["omnikvquant", "turboquant"], default="omnikvquant"
    )
    parser.add_argument("--data", type=Path, default=ROOT / "json/worldsense.json")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rotations", type=Path)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--attn", choices=["flash_attention_2", "sdpa"], default="flash_attention_2"
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--video-max-pixels", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_samples < 0 or args.max_new_tokens < 1:
        parser.error("max-samples must be >= 0 and max-new-tokens must be >= 1")
    if args.output.exists():
        parser.error(f"Output already exists; choose a new path: {args.output}")
    items = load_items(args.data)
    if args.max_samples:
        items = items[: args.max_samples]
    paths = [video_path(item, args.video_dir) for item in items]
    import torch
    from tqdm import tqdm
    from model import OmniKVQuantModel
    from inference.inputs import (
        seed_everything,
        build_conversation,
        build_inputs_cpu,
        move_inputs_to_model,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("WorldSense inference requires an NVIDIA CUDA GPU")
    seed_everything(args.seed)
    adapter = OmniKVQuantModel.from_pretrained(
        args.model,
        method=args.method,
        rotations_path=args.rotations,
        cache_dir=args.cache_dir,
        attn_implementation=args.attn,
    )
    model, processor = (adapter.model, adapter.processor)
    run_config = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    run_config["manifest_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = invalid = 0
    with args.output.open("x", encoding="utf-8") as output, torch.inference_mode():
        for item, path in tqdm(zip(items, paths), total=len(items)):
            conversation = build_conversation(
                str(path),
                item["prompt"],
                fps=args.video_fps,
                max_pixels=args.video_max_pixels,
            )
            inputs = move_inputs_to_model(
                build_inputs_cpu(processor, conversation), model
            )
            prompt_len = inputs["input_ids"].shape[1]
            ids, cache_stats = adapter.generate(
                inputs, max_new_tokens=args.max_new_tokens
            )
            text = processor.batch_decode(
                ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            prediction = parse_choice(text)
            ok = prediction == item["answer"]
            correct += int(ok)
            invalid += int(prediction is None)
            row = dict(
                id=item["id"],
                video=item["video"],
                answer=item["answer"],
                prediction=prediction,
                text=text,
                correct=ok,
                prompt_len=prompt_len,
                config=run_config,
                kv_cache=cache_stats,
            )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            del inputs, ids
    print(
        f"Accuracy: {100 * correct / len(items):.2f}% ({correct}/{len(items)}); invalid: {invalid}"
    )


if __name__ == "__main__":
    main()
