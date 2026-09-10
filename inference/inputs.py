import os
import torch
from qwen_omni_utils import process_mm_info


def seed_everything(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


_SYS = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please analyze the video carefully and select the most appropriate answer from the given options."
_LONG_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "video_grid_thw",
        "image_grid_thw",
        "feature_attention_mask",
        "position_ids",
        "cache_position",
        "rope_deltas",
        "labels",
        "audio_feature_lengths",
    }
)


def build_conversation(video_path: str, prompt: str, *, max_pixels=None, fps=None):
    prompt = prompt.replace("<image>", "").replace("<video>", "").strip()
    vid = {"type": "video", "video": video_path}
    if max_pixels is not None:
        vid["max_pixels"] = int(max_pixels)
    if fps is not None:
        vid["fps"] = fps
    return [
        {"role": "system", "content": [{"type": "text", "text": _SYS}]},
        {"role": "user", "content": [vid, {"type": "text", "text": prompt}]},
    ]


def build_inputs_cpu(processor, conversation, *, add_generation_prompt=True):
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=add_generation_prompt, tokenize=False
    )
    audios, _, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=None,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    return dict(inputs)


def move_inputs_to_model(inputs, model):
    inp = {}
    for k, v in inputs.items():
        if not torch.is_tensor(v):
            inp[k] = v
        elif k in _LONG_KEYS:
            inp[k] = v.to(model.device, dtype=torch.long)
        elif k == "input_features":
            inp[k] = v.to(model.device, dtype=torch.float32)
        elif k in ("pixel_values", "pixel_values_videos"):
            inp[k] = v.to(model.device).to(model.dtype)
        else:
            inp[k] = v.to(model.device)
    return inp
