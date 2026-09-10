from collections import defaultdict
from contextlib import contextmanager
import math
from typing import Optional
import torch
from transformers.cache_utils import DynamicCache
from model.codec.modality import TEXT, AUDIO, VIDEO
from model.codec.turboquant import hadamard_cpu
from inference.inputs import build_conversation, build_inputs_cpu

MODALITIES = (TEXT, AUDIO, VIDEO)


def add_matrix(store: dict, key: tuple, value: torch.Tensor) -> None:
    value = value.detach().double().cpu()
    if key not in store:
        store[key] = value
    else:
        store[key].add_(value)


def even_sample(indices: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0 or indices.numel() <= limit:
        return indices
    picks = torch.linspace(0, indices.numel() - 1, limit, device=indices.device)
    return indices.index_select(0, picks.round().long())


def rotation_from_covariance(covariance: torch.Tensor) -> torch.Tensor:
    covariance = covariance.double()
    covariance = 0.5 * (covariance + covariance.T)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis_t = eigenvectors.flip(1).T
    return (hadamard_cpu(covariance.shape[0]).double() @ basis_t).float().contiguous()


class CalibrationCollector(DynamicCache):
    def __init__(self, *, group: int, query_limit: int = 4):
        super().__init__()
        self.group = int(group)
        self.query_limit = int(query_limit)
        self.tags: Optional[torch.Tensor] = None
        self.query_mask: Optional[torch.Tensor] = None
        self.pending: dict[int, list] = {}
        self.v_covariances: dict = {}
        self.v_token_counts = defaultdict(int)
        self.completed_sample_ids: list[str] = []

    def reset_for_clip(self, tags: torch.Tensor, query_mask: torch.Tensor) -> None:
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0
        self.pending.clear()
        self.tags = tags.detach().cpu()
        self.query_mask = query_mask.detach().bool().cpu()

    def observe_qk(self, query: torch.Tensor, key: torch.Tensor, layer: int) -> None:
        if self.tags is None or query.shape[-2] <= 1:
            return
        q = query[0].detach().float()
        k = key[0].detach().float()
        length, dim = (q.shape[-2], q.shape[-1])
        caption_mask = self.query_mask[:length].to(q.device)
        tags = self.tags[:length].to(q.device)
        caption_positions = caption_mask.nonzero(as_tuple=True)[0]
        prefix_end = (
            int(caption_positions.min()) if caption_positions.numel() else length
        )
        prefix = torch.arange(length, device=q.device) < prefix_end
        groups = tuple(((tags == modality) & prefix for modality in MODALITIES))
        groups += (caption_mask,)
        sampled = []
        for selected in groups:
            index = selected.nonzero(as_tuple=True)[0]
            if index.numel():
                sampled.append(even_sample(index, self.query_limit))
        if not sampled:
            return
        qindex = torch.cat(sampled).sort().values
        key_positions = torch.arange(length, device=q.device)
        records = []
        for kv_head in range(k.shape[0]):
            qblock = q[kv_head * self.group : (kv_head + 1) * self.group].index_select(
                1, qindex
            )
            logits = torch.matmul(qblock, k[kv_head].T) / math.sqrt(dim)
            logits.masked_fill_(
                key_positions[None, None, :] > qindex[None, :, None], -torch.inf
            )
            probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
            records.append((kv_head, probabilities))
        self.pending[layer] = records

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if (
            key_states is not None
            and key_states.shape[-2] > 1
            and (self.tags is not None)
        ):
            length = key_states.shape[-2]
            tags = self.tags[:length].to(key_states.device)
            values = value_states[0].detach().float()
            records = self.pending.pop(layer_idx, [])
            for kv_head, probabilities in records:
                for modality in MODALITIES:
                    index = (tags == modality).nonzero(as_tuple=True)[0]
                    if not index.numel():
                        continue
                    pm = probabilities.index_select(-1, index)
                    vm = values[kv_head].index_select(0, index)
                    v_weight = torch.zeros(index.numel(), device=vm.device)
                    for local_head in range(self.group):
                        p2 = pm[local_head].square()
                        v_weight.add_(p2.sum(0))
                    cell = (int(layer_idx), int(kv_head), int(modality))
                    add_matrix(
                        self.v_covariances, cell, vm.T @ (vm * v_weight[:, None])
                    )
                    self.v_token_counts[cell] += int(index.numel())
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def caption_text(item: dict) -> str:
    for key in ("caption", "answer", "reference"):
        value = str(item.get(key, "") or "").strip()
        if value:
            return value
    for turn in item.get("conversations", []):
        if turn.get("from") in ("gpt", "assistant"):
            value = str(turn.get("value", "") or "").strip()
            if value:
                return value
    raise ValueError(f"missing calibration caption: id={item.get('id')!r}")


def prepare_inputs_cpu(processor, item: dict) -> dict:
    prompt = "Describe the video in detail."
    for turn in item.get("conversations", []):
        if turn.get("from") in ("human", "user"):
            prompt = str(turn.get("value", prompt))
            prompt = prompt.replace("<image>\n", "").replace("<video>\n", "")
            break
    conversation = build_conversation(item["video"], prompt)
    tokenizer = processor.tokenizer
    prefix_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True
    )
    conversation.append(
        {"role": "assistant", "content": [{"type": "text", "text": caption_text(item)}]}
    )
    full_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=False, tokenize=True
    )
    common = 0
    for left, right in zip(prefix_ids, full_ids):
        if left != right:
            break
        common += 1
    assistant_suffix = len(full_ids) - common
    if assistant_suffix <= 0:
        raise RuntimeError("teacher-forced caption produced no query tokens")
    inputs = build_inputs_cpu(processor, conversation, add_generation_prompt=False)
    length = inputs["input_ids"].shape[-1]
    target_start = length - assistant_suffix
    query_mask = torch.zeros(length, dtype=torch.bool)
    query_mask[max(0, target_start - 1) : length - 1] = True
    return {"inputs": inputs, "query_mask": query_mask}


@contextmanager
def observe_attention(collector, num_layers):
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as modeling

    original = modeling.apply_multimodal_rotary_pos_emb
    current = 0

    def wrapped(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        nonlocal current
        qe, ke = original(q, k, cos, sin, mrope_section, unsqueeze_dim)
        collector.observe_qk(qe, ke, current % num_layers)
        current += 1
        return (qe, ke)

    modeling.apply_multimodal_rotary_pos_emb = wrapped
    try:
        yield
        if current != num_layers:
            raise RuntimeError(
                f"Expected {num_layers} attention layers; observed {current}"
            )
    finally:
        modeling.apply_multimodal_rotary_pos_emb = original
