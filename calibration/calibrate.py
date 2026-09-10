#!/usr/bin/env python
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_manifest(path, video_dir):
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Calibration manifest must be a nonempty JSON list")
    seen_ids, seen_classes = (set(), set())
    for item in items:
        identity = str(item["id"])
        label = str(item["class_label"])
        if identity in seen_ids or label in seen_classes:
            raise ValueError("Calibration requires one unique clip per class")
        if item.get("split") != "train":
            raise ValueError("Calibration uses VGGSound training clips only")
        seen_ids.add(identity)
        seen_classes.add(label)
        video = Path(item["video"])
        if not video.is_absolute():
            video = video_dir / video
        if not video.is_file():
            raise FileNotFoundError(video)
        item["video"] = str(video)
    return items


def save_checkpoint(collector, text_config, output, metadata):
    import torch
    from calibration.collector import rotation_from_covariance

    dim = text_config.hidden_size // text_config.num_attention_heads
    rotations = {}
    for layer in range(text_config.num_hidden_layers):
        for head in range(text_config.num_key_value_heads):
            for modality in (0, 1, 2):
                cell = (layer, head, modality)
                covariance = collector.v_covariances.get(cell)
                if covariance is None or covariance.shape != (dim, dim):
                    raise ValueError(f"Missing calibration covariance: {cell}")
                if not torch.isfinite(covariance).all() or covariance.trace() <= 0:
                    raise ValueError(f"Invalid calibration covariance: {cell}")
                rotations[cell] = rotation_from_covariance(covariance)
    payload = dict(
        rotations=rotations,
        meta={
            **metadata,
            "convention": "vllm_normalized_post_rope_hvt_v1",
            "method": "attention_aware",
            "tensor": "v",
            "sharing": "per_modality",
            "query_source": "balanced_prefill_modalities_plus_teacher_forced_caption_text",
            "calibration_clips": len(collector.completed_sample_ids),
            "sample_ids": collector.completed_sample_ids,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(rotations)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate OmniKVQuant V rotations on a fixed VGGSound training manifest."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("vggsound_309.json")
    )
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    items = load_manifest(args.manifest, args.video_dir)
    import torch
    from tqdm import tqdm
    from calibration.collector import (
        CalibrationCollector,
        prepare_inputs_cpu,
        observe_attention,
        caption_text,
    )
    from inference.inputs import seed_everything, move_inputs_to_model
    from model.modeling_qwen2_5_omni import load_backbone
    from model.codec.modality import modality_tags

    if not torch.cuda.is_available():
        raise RuntimeError("VGGSound calibration requires a CUDA GPU")
    for item in items:
        caption_text(item)
    seed_everything(42)
    torch.set_num_threads(8)
    model, processor = load_backbone(args.model, cache_dir=args.cache_dir)
    text_config = model.config.thinker_config.text_config
    collector = CalibrationCollector(
        group=text_config.num_attention_heads // text_config.num_key_value_heads
    )
    with torch.inference_mode():
        for item in tqdm(items, desc="Calibrating V rotations"):
            try:
                prepared = prepare_inputs_cpu(processor, item)
                inputs = move_inputs_to_model(prepared["inputs"], model)
                collector.reset_for_clip(
                    modality_tags(inputs["input_ids"], model), prepared["query_mask"]
                )
                with observe_attention(collector, text_config.num_hidden_layers):
                    model.generate(
                        **inputs,
                        use_audio_in_video=True,
                        return_audio=False,
                        thinker_max_new_tokens=1,
                        do_sample=False,
                        thinker_past_key_values=collector,
                    )
                collector.completed_sample_ids.append(str(item["id"]))
                del inputs, prepared
            except Exception as exc:
                raise RuntimeError(
                    f"Calibration failed on clip {item['id']}; no checkpoint was published"
                ) from exc
    count = save_checkpoint(
        collector,
        text_config,
        args.output,
        {
            "model": args.model,
            "seed": 42,
            "query_limit": 4,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__),
        },
    )
    print(f"Saved {count} V rotation matrices from {len(items)} clips to {args.output}")


if __name__ == "__main__":
    main()
