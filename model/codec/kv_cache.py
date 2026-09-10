from dataclasses import dataclass
import torch
from transformers.cache_utils import DynamicCache
from .modality import MODALITY_NAMES, TEXT
from .packed_cache import CODEC_LOCAL_LM, CODEC_LM, PackedLayer, pack_unsigned
from .turboquant import hadamard, lloyd_max


@dataclass(frozen=True)
class OmniKVQuantConfig:
    method: str
    rotations_v: dict | None = None

    def __post_init__(self):
        if self.method not in ("omnikvquant", "turboquant"):
            raise ValueError(f"Unsupported method: {self.method}")
        if self.method == "omnikvquant" and (not self.rotations_v):
            raise ValueError("OmniKVQuant requires calibrated V rotations")
        if self.method == "turboquant" and self.rotations_v is not None:
            raise ValueError("TurboQuant uses the shared Hadamard for K and V")


class OmniKVQuantCache(DynamicCache):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._tags = None
        self._rots = {}
        self._packed = {"k": {}, "v": {}}
        self._packed_pending = {}
        self._packed_seq_lens = {}
        self._packed_decode_current = {}
        self._elems = {"k": {}, "v": {}}
        self._bits = {"k": {}, "v": {}}
        self._packed_codebooks = {}
        self._packed_fused_store = False
        self._packed_fused_decode = False
        self._packed_fused_scratch_bytes = 0

    def set_prefill_tags(self, tags):
        tags = tags.to(device="cpu", dtype=torch.uint8)
        if tags.ndim != 1 or any((int(m) not in (0, 1, 2) for m in tags.unique())):
            raise ValueError("WorldSense cache requires text/audio/video tags")
        self._tags = tags

    def _chunk_tags(self, offset, length):
        tags = torch.full((length,), TEXT, dtype=torch.uint8)
        if self._tags is not None and offset < len(self._tags):
            count = min(length, len(self._tags) - offset)
            tags[:count] = self._tags[offset : offset + count]
        return tags

    def _shared_rot(self, layer_idx, heads, dim, device, which="k"):
        key = ("shared", layer_idx, which, str(device))
        if key not in self._rots:
            self._rots[key] = (
                hadamard(dim).expand(heads, dim, dim).contiguous().to(device)
            )
        return self._rots[key]

    def _rot(self, layer_idx, mod, heads, dim, device, which="k"):
        if which == "k" or self.cfg.method == "turboquant":
            return self._shared_rot(layer_idx, heads, dim, device, which)
        key = (which, layer_idx, mod, str(device))
        if key not in self._rots:
            self._rots[key] = torch.stack(
                [
                    self.cfg.rotations_v[layer_idx, head, mod].float()
                    for head in range(heads)
                ]
            ).to(device)
        return self._rots[key]

    def _pack_append(self, x, offset, layer_idx, which):
        batch, heads, tokens, dim = x.shape
        if layer_idx not in self._packed[which]:
            self._packed[which][layer_idx] = PackedLayer(
                batch, heads, dim, x.dtype, x.device
            )
        layer = self._packed[which][layer_idx]
        local_keys = which == "k" and self.cfg.method == "omnikvquant"
        codec = CODEC_LOCAL_LM if local_keys else CODEC_LM
        shared = which == "k" or self.cfg.method == "turboquant"
        values = norms = None
        if shared:
            source = x.float()
            norms = source.norm(dim=-1).clamp_min(1e-12)
            rotation = self._shared_rot(layer_idx, heads, dim, x.device, which)
            values = torch.einsum(
                "bhtd,hcd->bhtc", source / norms.unsqueeze(-1), rotation
            )
        tags = self._chunk_tags(offset, tokens)
        for mod in tags.unique().tolist():
            indices = (tags == mod).nonzero(as_tuple=True)[0]
            device_indices = indices.to(x.device)
            positions = offset + indices
            if shared:
                sub = values.index_select(2, device_indices)
                norm = norms.index_select(2, device_indices)
            else:
                sub = x.index_select(2, device_indices).float()
                norm = sub.norm(dim=-1).clamp_min(1e-12)
                rotation = self._rot(layer_idx, mod, heads, dim, x.device, which)
                sub = torch.einsum("bhtd,hcd->bhtc", sub / norm.unsqueeze(-1), rotation)
            group = layer.group(mod, codec)
            if local_keys:
                group.append_local_lm(sub, norm, positions)
            else:
                boundaries = lloyd_max(dim)[1].to(x.device)
                if x.is_cuda:
                    from .triton_store import pack_lm

                    index, norm16 = pack_lm(sub, norm, boundaries)
                else:
                    index = pack_unsigned(
                        torch.bucketize(sub.contiguous(), boundaries, right=True)
                    )
                    norm16 = norm.to(torch.float16)
                group.append(index, norm16, positions)
            self._packed_fused_store |= x.is_cuda
            elements = batch * heads * indices.numel() * dim
            effective = 2.0 + 16.0 / dim + (8.5 / 32 if local_keys else 0)
            self._elems[which][mod] = self._elems[which].get(mod, 0) + elements
            self._bits[which][mod] = (
                self._bits[which].get(mod, 0) + elements * effective
            )

    def _pack_append_pair(self, key, value, offset, layer_idx):
        self._pack_append(key, offset, layer_idx, "k")
        self._pack_append(value, offset, layer_idx, "v")

    def quantize_pending(self, layer_idx):
        pending = self._packed_pending.pop(layer_idx, None)
        if pending is not None:
            key, value, offset = pending
            self._pack_append_pair(key, value, offset, layer_idx)

    def current_token(self, layer_idx):
        return self._packed_decode_current.get(layer_idx, (None, None, None))

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape != value_states.shape or key_states.shape[0] != 1:
            raise ValueError("Expected matching K/V tensors with batch size one")
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0, device=key_states.device))
            self.value_cache.append(torch.empty(0, device=key_states.device))
        length = key_states.shape[2]
        old = self.get_seq_length(layer_idx)
        if (
            layer_idx in self._packed_pending
            or layer_idx in self._packed_decode_current
        ):
            raise RuntimeError("Previous attention write has not been finalized")
        if layer_idx == 0:
            self._seen_tokens += length
        self._packed_seq_lens[layer_idx] = old + length
        if length > 1:
            if old:
                raise ValueError(
                    "This runner supports one prefill followed by one-token decode"
                )
            self._packed_pending[layer_idx] = (key_states, value_states, old)
            return (key_states, value_states)
        if length != 1:
            raise ValueError("Cannot append an empty KV chunk")
        positions = torch.tensor([old], device=key_states.device, dtype=torch.int32)
        self._packed_decode_current[layer_idx] = (key_states, value_states, positions)
        if (cache_kwargs or {}).get("omnikvquant_packed_fused", False):
            return (key_states, value_states)
        return (self._unpack_layer(layer_idx, "k"), self._unpack_layer(layer_idx, "v"))

    def finalize_decode(self, layer_idx):
        current = self._packed_decode_current.pop(layer_idx, None)
        if current is not None:
            key, value, position = current
            self._pack_append_pair(key, value, int(position[0]), layer_idx)

    def get_seq_length(self, layer_idx=0):
        return self._packed_seq_lens.get(layer_idx or 0, 0)

    def _unpack_layer(self, layer_idx, which):
        layer = self._packed[which][layer_idx]
        total = self.get_seq_length(layer_idx)
        output = torch.empty(
            layer.batch,
            layer.heads,
            total,
            layer.dim,
            device=layer.device,
            dtype=torch.float32,
        )
        filled = torch.zeros(total, device=layer.device, dtype=torch.bool)
        for mod, group in layer.groups.items():
            values, positions = group.unpack_rotated()
            rotation = self._rot(
                layer_idx, mod, layer.heads, layer.dim, layer.device, which
            )
            restored = torch.einsum("bhtc,hcd->bhtd", values, rotation)
            output.index_copy_(2, positions, restored)
            filled[positions] = True
        key, value, positions = self.current_token(layer_idx)
        if key is not None:
            output.index_copy_(
                2, positions.long(), (key if which == "k" else value).float()
            )
            filled[positions.long()] = True
        if not filled.all():
            raise RuntimeError("Packed cache contains missing token positions")
        return output.to(layer.dtype)

    def summary(self):
        result = {
            "method": self.cfg.method,
            "bits_k": 2,
            "bits_v": 2,
            "k_local_group_size": 32 if self.cfg.method == "omnikvquant" else 0,
            "k_local_metadata": "mxfp4" if self.cfg.method == "omnikvquant" else None,
        }
        elements = bits = allocated = live = fp16 = 0
        for which in ("k", "v"):
            for mod, count in self._elems[which].items():
                used = self._bits[which][mod]
                result[f"bits_{which}_{MODALITY_NAMES[mod]}"] = round(used / count, 3)
                elements += count
                bits += used
            for layer in self._packed[which].values():
                allocated += layer.storage_bytes()
                live += layer.storage_bytes(live=True)
                fp16 += (
                    layer.batch
                    * layer.heads
                    * sum((g.length for g in layer.groups.values()))
                    * layer.dim
                    * 2
                )
        if elements:
            result.update(
                bits_avg=round(bits / elements, 3),
                payload_avg=2.0,
                mem_ratio_vs_fp16=round(bits / (elements * 16), 4),
            )
        result.update(
            packed_cache_bytes=allocated,
            packed_cache_live_bytes=live,
            fp16_cache_bytes=fp16,
            packed_fused_store=int(self._packed_fused_store),
            packed_fused_decode=int(self._packed_fused_decode),
            packed_fused_scratch_bytes=self._packed_fused_scratch_bytes,
        )
        if fp16:
            result.update(
                packed_mem_ratio_vs_fp16=round(allocated / fp16, 4),
                packed_live_ratio_vs_fp16=round(live / fp16, 4),
                packed_bits_per_elem_actual=round(allocated * 16 / fp16, 3),
            )
        return result
