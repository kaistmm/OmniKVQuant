import math
import types
from typing import Optional
import torch
from .kv_cache import OmniKVQuantCache


def _repeat_kv_block(x: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return x
    return x.repeat_interleave(num_key_value_groups, dim=1)


def _mask_block(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    start: int,
    end: int,
    total: int,
) -> torch.Tensor:
    if attention_mask is None:
        return logits
    if attention_mask.ndim == 2:
        keep = attention_mask[:, start:end].to(torch.bool)
        return logits.masked_fill(~keep[:, None, None, :], float("-inf"))
    if attention_mask.ndim == 4:
        add = attention_mask[..., -1:, start:end].float()
        blocked = add < -10000.0
        return (logits + add).masked_fill(blocked, float("-inf"))
    raise ValueError(
        f"unsupported attention_mask rank {attention_mask.ndim}; expected 2 or 4"
    )


def online_decode_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    num_key_value_groups: int,
    attention_mask: Optional[torch.Tensor] = None,
    block_size: int = 256,
) -> torch.Tensor:
    if query_states.ndim != 4 or query_states.shape[-2] != 1:
        raise ValueError("online_decode_attention only supports one-token decode")
    if key_states.shape != value_states.shape:
        raise ValueError("key_states and value_states must have identical shapes")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    expected_hq = key_states.shape[1] * num_key_value_groups
    if query_states.shape[1] != expected_hq:
        raise ValueError(
            f"GQA head mismatch: Hq={query_states.shape[1]} vs expected {expected_hq}"
        )
    bsz, hq, _, dim = query_states.shape
    total = key_states.shape[-2]
    if total == 0:
        raise ValueError("cannot attend to an empty KV cache")
    q = query_states.float()
    running_max = torch.full((bsz, hq, 1, 1), float("-inf"), device=q.device)
    running_sum = torch.zeros_like(running_max)
    accumulator = torch.zeros((bsz, hq, 1, dim), dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(dim)
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        kb = _repeat_kv_block(
            key_states[:, :, start:end, :], num_key_value_groups
        ).float()
        vb = _repeat_kv_block(
            value_states[:, :, start:end, :], num_key_value_groups
        ).float()
        logits = torch.matmul(q, kb.transpose(-2, -1)) * scale
        logits = _mask_block(logits, attention_mask, start, end, total)
        block_max = logits.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, block_max)
        finite_new = torch.isfinite(new_max)
        old_scale = torch.where(
            torch.isfinite(running_max) & finite_new,
            torch.exp(running_max - new_max),
            torch.zeros_like(new_max),
        )
        probs = torch.where(
            torch.isfinite(logits) & finite_new,
            torch.exp(logits - new_max),
            torch.zeros_like(logits),
        )
        accumulator = accumulator * old_scale + torch.matmul(probs, vb)
        running_sum = running_sum * old_scale + probs.sum(dim=-1, keepdim=True)
        running_max = new_max
    output = accumulator / running_sum.clamp_min(1e-20)
    return output.to(query_states.dtype)


def install_attention(model) -> int:
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as modeling

    count = 0
    for module in model.modules():
        if not isinstance(module, modeling.Qwen2_5OmniAttention):
            continue
        if hasattr(module, "_omnikvquant_original_forward"):
            continue
        module._omnikvquant_original_forward = module.forward
        module.forward = types.MethodType(_attention_forward, module)
        count += 1
    if count == 0:
        raise RuntimeError("no Qwen2_5OmniAttention thinker modules found to patch")
    return count


def _attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
):
    original = self._omnikvquant_original_forward
    if not isinstance(past_key_value, OmniKVQuantCache):
        return original(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    if self.training or output_attentions:
        raise ValueError(
            "Packed attention supports inference without attention-map output"
        )
    batch, length, _ = hidden_states.shape
    if length > 1:
        outputs = original(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        past_key_value.quantize_pending(self.layer_idx)
        return outputs
    query = (
        self.q_proj(hidden_states)
        .view(batch, length, -1, self.head_dim)
        .transpose(1, 2)
    )
    key = (
        self.k_proj(hidden_states)
        .view(batch, length, -1, self.head_dim)
        .transpose(1, 2)
    )
    value = (
        self.v_proj(hidden_states)
        .view(batch, length, -1, self.head_dim)
        .transpose(1, 2)
    )
    if position_embeddings is None:
        raise ValueError("Qwen decode requires position embeddings")
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as modeling

    cos, sin = position_embeddings
    query, key = modeling.apply_multimodal_rotary_pos_emb(
        query, key, cos, sin, self.rope_scaling["mrope_section"]
    )
    packed_cuda = query.is_cuda
    key, value = past_key_value.update(
        key,
        value,
        self.layer_idx,
        {
            "cos": cos,
            "sin": sin,
            "cache_position": cache_position,
            "omnikvquant_packed_fused": packed_cuda,
        },
    )
    dtype = self.q_proj.weight.dtype
    if query.dtype == torch.float32 and dtype != torch.float32:
        query, key, value = (query.to(dtype), key.to(dtype), value.to(dtype))
    if getattr(self.config, "use_sliding_window", False):
        raise ValueError(
            "This release requires Qwen's full-context attention configuration"
        )
    if packed_cuda:
        from .triton_packed_attention import packed_decode_attention

        result = packed_decode_attention(
            query,
            past_key_value,
            self.layer_idx,
            num_key_value_groups=self.num_key_value_groups,
            attention_mask=attention_mask,
        )
    else:
        result = online_decode_attention(
            query,
            key,
            value,
            num_key_value_groups=self.num_key_value_groups,
            attention_mask=attention_mask,
        )
    past_key_value.finalize_decode(self.layer_idx)
    result = result.transpose(1, 2).contiguous().reshape(batch, length, -1)
    return (self.o_proj(result), None, past_key_value)
