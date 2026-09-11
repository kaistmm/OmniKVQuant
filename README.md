# OmniKVQuant: KV Cache Quantization for Omni-LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2609.11582-b31b1b.svg)](https://arxiv.org/abs/2609.11582)

Official implementation of **OmniKVQuant**, a training-free KV-cache quantization
method for Omni-LLMs

This release includes the calibrated value rotations, packed 2-bit cache, fused Triton decode kernels, and the TurboQuant baseline.

## Installation

Run commands from the repository root.

```bash
git clone https://github.com/kaistmm/OmniKVQuant.git
cd OmniKVQuant

conda create -n omnikvquant python=3.10 -y
conda activate omnikvquant
conda install -c conda-forge ffmpeg -y

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install packaging ninja
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

## WorldSense data

Download the videos from [WorldSense](https://huggingface.co/datasets/honglyhly/WorldSense).
Set `--video-dir` to the directory containing the `.mp4` files with audio.
The included `json/worldsense.json` contains 799 questions and uses video filenames
relative to this directory.

```text
/path/to/WorldSense/
  KZsaltBw.mp4
  ...
```

## Inference

OmniKVQuant uses the included `calibration/3b/v_rotation.pt`.

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_worldsense.py \
  --method omnikvquant \
  --video-dir /path/to/WorldSense \
  --output outputs/worldsense_omnikvquant.jsonl
```

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_worldsense.py \
  --method turboquant \
  --video-dir /path/to/WorldSense \
  --output outputs/worldsense_turboquant.jsonl
```

Use a new output filename for each run; existing files are not overwritten.

## Evaluation

```bash
python eval/eval_worldsense.py outputs/worldsense_omnikvquant.jsonl
python eval/eval_worldsense.py outputs/worldsense_turboquant.jsonl
```

A complete run reports `evaluated: 799` and `missing: 0`.

## Calibration

The supplied checkpoint is ready for inference. To generate a new checkpoint,
use the included 309-clip VGGSound list:

```bash
CUDA_VISIBLE_DEVICES=0 python calibration/calibrate.py \
  --video-dir /path/to/VGGSound/video \
  --output calibration/3b/new_v_rotation.pt
```

To create a calibration list from your own VGGSound training annotations:

```bash
python calibration/prepare_manifest.py \
  --source /path/to/vggsound_training_annotations.jsonl \
  --output calibration/my_vggsound.json
```

The source accepts JSON or JSONL with records in this format:

```json
{"id": "DQC78JSBJoo_000000", "video": "DQC78JSBJoo_000000.mp4", "split": "train", "class_label": "slot machine", "caption": "slot machine"}
```

The script selects one training clip per class. Calibrate with the resulting list:

```bash
CUDA_VISIBLE_DEVICES=0 python calibration/calibrate.py \
  --manifest calibration/my_vggsound.json \
  --video-dir /path/to/VGGSound/video \
  --output calibration/3b/custom_v_rotation.pt
```

Run inference with the generated checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python inference/inference_worldsense.py \
  --method omnikvquant \
  --video-dir /path/to/WorldSense \
  --rotations calibration/3b/custom_v_rotation.pt \
  --output outputs/worldsense_custom.jsonl
```

For the included-list calibration command, use
`--rotations calibration/3b/new_v_rotation.pt` instead.

## Acknowledgments

This implementation builds on [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni),
[Transformers](https://github.com/huggingface/transformers), and the
[vLLM](https://github.com/vllm-project/vllm) TurboQuant primitives.

## Citation

```bibtex
@article{yoo2026omnikvquant,
  title={{OmniKVQuant: KV Cache Quantization for Omni-LLMs}},
  author={Yoo, Suho and Ok, Hyunjong and Choi, Jongmin and Jung, Jihoo and Chung, Joon Son},
  journal={arXiv preprint arXiv:2609.11582},
  year={2026}
}
```

## License

[Apache-2.0](LICENSE). Third-party attribution is listed in [NOTICE](NOTICE).
