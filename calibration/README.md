# Calibration

Run these commands from the repository root after [installation](../README.md#installation).
Set `--video-dir` to the directory containing the VGGSound `.mp4` clips with audio.

## Generate a checkpoint

The default input is `calibration/vggsound_309.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python calibration/calibrate.py \
  --video-dir /path/to/VGGSound/video \
  --output calibration/3b/new_v_rotation.pt
```

## Prepare your own clip list

```bash
python calibration/prepare_manifest.py \
  --source /path/to/vggsound_training_annotations.jsonl \
  --output calibration/my_vggsound.json
```

The source is JSON or JSONL containing records like:

```json
{"id": "DQC78JSBJoo_000000", "video": "DQC78JSBJoo_000000.mp4", "split": "train", "class_label": "slot machine", "caption": "slot machine"}
```

One training clip is selected per class. Video filenames resolve against `--video-dir`.

```bash
CUDA_VISIBLE_DEVICES=0 python calibration/calibrate.py \
  --manifest calibration/my_vggsound.json \
  --video-dir /path/to/VGGSound/video \
  --output calibration/3b/custom_v_rotation.pt
```

## Use the checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_worldsense.py \
  --method omnikvquant \
  --video-dir /path/to/WorldSense \
  --rotations calibration/3b/custom_v_rotation.pt \
  --output outputs/worldsense_custom.jsonl
```

For the default-list run, use `--rotations calibration/3b/new_v_rotation.pt`.
All listed clips are required. Use new filenames for generated lists and checkpoints;
existing outputs are not overwritten.
