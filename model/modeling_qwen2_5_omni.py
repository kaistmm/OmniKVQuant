from pathlib import Path
import torch
from .codec.kv_cache import OmniKVQuantCache, OmniKVQuantConfig
from .codec.modality import modality_tags
from .codec.attention import install_attention

ROOT = Path(__file__).resolve().parents[1]


def make_cache_config(method, *, rotations=None):
    return OmniKVQuantConfig(method=method, rotations_v=rotations)


def load_value_rotations(path, text_config):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("meta", {}).get("convention")
        != "vllm_normalized_post_rope_hvt_v1"
    ):
        raise ValueError("Incompatible value-rotation convention")
    rotations = checkpoint["rotations"]
    dim = text_config.hidden_size // text_config.num_attention_heads
    for layer in range(text_config.num_hidden_layers):
        for head in range(text_config.num_key_value_heads):
            for modality in (0, 1, 2):
                key = (layer, head, modality)
                matrix = rotations.get(key)
                if matrix is None or matrix.shape != (dim, dim):
                    raise ValueError(f"Missing or incompatible rotation: {key}")
                if not torch.isfinite(matrix).all():
                    raise ValueError(f"Non-finite rotation: {key}")
    return rotations


_THINKER_PREFILL_KEYS = frozenset(
    {
        "input_features",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "feature_attention_mask",
        "audio_feature_lengths",
        "video_second_per_grid",
    }
)


def prefill_mcq_prefix(model, inp: dict, cache) -> int:
    input_ids = inp["input_ids"]
    prompt_len = int(input_ids.shape[1])
    if prompt_len < 2:
        raise ValueError("MCQ continuation mode needs at least two prompt tokens")
    prefix_len = prompt_len - 1
    kwargs = {k: inp[k] for k in _THINKER_PREFILL_KEYS if k in inp}
    kwargs["input_ids"] = input_ids[:, :prefix_len]
    if "attention_mask" in inp:
        kwargs["attention_mask"] = inp["attention_mask"][:, :prefix_len]
    kwargs.update(
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
        use_audio_in_video=True,
        cache_position=torch.arange(prefix_len, device=input_ids.device),
    )
    model.thinker(**kwargs)
    return prefix_len


def load_backbone(
    model_name="Qwen/Qwen2.5-Omni-3B",
    *,
    config=None,
    cache_dir=None,
    attn_implementation="flash_attention_2",
):
    from transformers import (
        Qwen2_5OmniConfig,
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    if config is None:
        config = Qwen2_5OmniConfig.from_pretrained(model_name, cache_dir=cache_dir)
    config.enable_audio_output = False
    model = (
        super(Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniForConditionalGeneration)
        .from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation=attn_implementation,
        )
        .eval()
    )
    model.speaker_map["Chelsie"] = {}
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    return (model, processor)


class OmniKVQuantModel:
    def __init__(self, model, processor, cache_config):
        self.model = model
        self.processor = processor
        self.cache_config = cache_config
        install_attention(model)

    @classmethod
    def from_pretrained(
        cls,
        model_name="Qwen/Qwen2.5-Omni-3B",
        *,
        method="omnikvquant",
        rotations_path=None,
        cache_dir=None,
        attn_implementation="flash_attention_2",
    ):
        from transformers import Qwen2_5OmniConfig

        config = Qwen2_5OmniConfig.from_pretrained(model_name, cache_dir=cache_dir)
        rotations = None
        if method == "omnikvquant":
            rotations = load_value_rotations(
                rotations_path or ROOT / "calibration/3b/v_rotation.pt",
                config.thinker_config.text_config,
            )
        cache_config = make_cache_config(method, rotations=rotations)
        model, processor = load_backbone(
            model_name,
            config=config,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
        )
        return cls(model, processor, cache_config)

    def new_cache(self, input_ids):
        cache = OmniKVQuantCache(self.cache_config)
        cache.set_prefill_tags(modality_tags(input_ids, self.model)[:-1])
        return cache

    @torch.inference_mode()
    def generate(self, inputs, *, max_new_tokens=1):
        cache = self.new_cache(inputs["input_ids"])
        kwargs = dict(
            use_audio_in_video=True,
            return_audio=False,
            thinker_max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        prefill_mcq_prefix(self.model, inputs, cache)
        kwargs["thinker_past_key_values"] = cache
        ids = self.model.generate(**inputs, **kwargs)
        return (ids, cache.summary())
