from __future__ import annotations
import torch

TEXT, AUDIO, VIDEO, IMAGE = (0, 1, 2, 3)
MODALITY_NAMES = {TEXT: "text", AUDIO: "audio", VIDEO: "video", IMAGE: "image"}


def _cfg_get(cfg, *names):
    for n in names:
        v = getattr(cfg, n, None)
        if v is not None:
            return v
    return None


def mm_token_ids(model) -> dict:
    cfg = getattr(model.config, "thinker_config", model.config)
    return {
        AUDIO: _cfg_get(cfg, "audio_token_id", "audio_token_index"),
        VIDEO: _cfg_get(cfg, "video_token_id", "video_token_index"),
        IMAGE: _cfg_get(cfg, "image_token_id", "image_token_index"),
    }


def modality_tags(input_ids: torch.Tensor, model) -> torch.Tensor:
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    tags = torch.zeros(ids.shape[0], dtype=torch.uint8)
    for mod, tok in mm_token_ids(model).items():
        if tok is not None:
            tags[(ids == tok).cpu()] = mod
    return tags
