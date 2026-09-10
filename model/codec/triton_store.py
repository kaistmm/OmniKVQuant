# SPDX-License-Identifier: Apache-2.0
import math
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None
if triton is not None:

    @triton.jit(do_not_specialize=("TOKENS", "OUT_CAPACITY", "TOKEN_OFFSET"))
    def _local_lm_pack_into_kernel(
        values_ptr,
        norms_ptr,
        boundaries_ptr,
        out_index_ptr,
        out_norm_ptr,
        metadata_min_ptr,
        metadata_span_ptr,
        TOKENS,
        OUT_CAPACITY,
        TOKEN_OFFSET,
        D: tl.constexpr,
        WIDTH: tl.constexpr,
    ):
        route_groups = tl.cdiv(TOKENS, 32)
        program = tl.program_id(0)
        head_batch = program // route_groups
        group = program - head_batch * route_groups
        token_offsets = tl.arange(0, 32)
        dim_offsets = tl.arange(0, D)
        source_tokens = group * 32 + token_offsets
        valid_tokens = source_tokens < TOKENS
        values = tl.load(
            values_ptr
            + (head_batch * TOKENS + source_tokens[:, None]) * D
            + dim_offsets[None, :],
            mask=valid_tokens[:, None],
            other=0.0,
        ).to(tl.float32)
        metadata_base = (head_batch * route_groups + group) * D + dim_offsets
        minimum = tl.load(metadata_min_ptr + metadata_base).to(tl.float32)
        span = tl.load(metadata_span_ptr + metadata_base).to(tl.float32)
        safe_span = tl.where(span == 0.0, 1.0, span)
        unit = tl.minimum(
            tl.maximum((values - minimum[None, :]) / safe_span[None, :], 0.0), 1.0
        )
        low = tl.zeros([32, D], dtype=tl.int32)
        high = tl.full([32, D], 4 - 1, dtype=tl.int32)
        for _ in range(2):
            middle = (low + high) // 2
            safe = tl.minimum(middle, 4 - 2)
            boundary = tl.load(boundaries_ptr + safe)
            go_right = unit >= boundary
            low = tl.where(go_right, middle + 1, low)
            high = tl.where(go_right, high, middle)
        index = tl.minimum(low, 4 - 1)
        output_tokens = TOKEN_OFFSET + group * 32 + token_offsets
        packed = tl.sum(
            index.reshape([32, D // 4, 4]) << (tl.arange(0, 4) * 2)[None, None, :],
            axis=2,
        )
        byte_offsets = tl.arange(0, D // 4)
        output_base = (head_batch * OUT_CAPACITY + output_tokens[:, None]) * WIDTH
        tl.store(
            out_index_ptr + output_base + byte_offsets[None, :],
            packed.to(tl.uint8),
            mask=valid_tokens[:, None],
        )
        norm = tl.load(
            norms_ptr + head_batch * TOKENS + source_tokens,
            mask=valid_tokens,
            other=0.0,
        )
        tl.store(
            out_norm_ptr + head_batch * OUT_CAPACITY + output_tokens,
            norm.to(tl.float16),
            mask=valid_tokens,
        )

    @triton.jit(do_not_specialize=("TOKENS",))
    def _lm_pack_kernel(
        values_ptr,
        norms_ptr,
        boundaries_ptr,
        out_index_ptr,
        out_norm_ptr,
        value_stride: tl.constexpr,
        index_stride: tl.constexpr,
        TOKENS,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        vector = tl.program_id(0)
        head = vector // TOKENS % HEADS
        offsets = tl.arange(0, BLOCK_D)
        valid = offsets < D
        values = tl.load(
            values_ptr + vector * value_stride + offsets, mask=valid, other=0.0
        ).to(tl.float32)
        low = tl.zeros([BLOCK_D], dtype=tl.int32)
        high = tl.full([BLOCK_D], 4 - 1, dtype=tl.int32)
        for _ in range(2):
            middle = (low + high) // 2
            safe = tl.minimum(middle, 4 - 2)
            boundary = tl.load(
                boundaries_ptr + head * (4 - 1) + safe, mask=valid, other=0.0
            )
            go_right = values >= boundary
            low = tl.where(go_right, middle + 1, low)
            high = tl.where(go_right, high, middle)
        index = tl.minimum(low, 4 - 1)
        shifts = tl.arange(0, 4) * 2
        packed = tl.sum(index.reshape([D // 4, 4]) << shifts[None, :], axis=1)
        packed_offsets = tl.arange(0, D // 4)
        tl.store(
            out_index_ptr + vector * index_stride + packed_offsets, packed.to(tl.uint8)
        )
        tl.store(out_norm_ptr + vector, tl.load(norms_ptr + vector).to(tl.float16))


def pack_local_lm_into(
    values,
    norms,
    boundaries,
    out_index,
    out_norm,
    *,
    token_offset,
    metadata_minimum,
    metadata_span,
):
    if triton is None or not values.is_cuda:
        raise RuntimeError("Packed CUDA writes require Triton")
    batch, heads, tokens, dim = values.shape
    groups = math.ceil(tokens / 32)
    expected = (batch, heads, groups, dim)
    if metadata_minimum.shape != expected or metadata_span.shape != expected:
        raise ValueError("Invalid local metadata shape")
    if token_offset % 32 or token_offset + tokens > out_index.shape[2]:
        raise ValueError("Invalid destination offset/capacity")
    values, norms = (values.contiguous(), norms.contiguous())
    boundaries = boundaries.to(values.device, dtype=torch.float32).contiguous()
    metadata_minimum = metadata_minimum.to(
        values.device, dtype=torch.float16
    ).contiguous()
    metadata_span = metadata_span.to(values.device, dtype=torch.float16).contiguous()
    width = dim // 4
    _local_lm_pack_into_kernel[batch * heads * groups,](
        values,
        norms,
        boundaries,
        out_index,
        out_norm,
        metadata_minimum,
        metadata_span,
        TOKENS=tokens,
        OUT_CAPACITY=out_index.shape[2],
        TOKEN_OFFSET=token_offset,
        D=dim,
        WIDTH=width,
        num_warps=8,
        num_stages=1,
    )


def pack_lm(values: torch.Tensor, norms: torch.Tensor, boundaries: torch.Tensor):
    if triton is None or not values.is_cuda:
        raise RuntimeError("Packed writes require CUDA Triton")
    dim = values.shape[-1]
    contiguous = values.contiguous()
    heads = values.shape[1]
    tokens = values.shape[2]
    if boundaries.ndim == 1:
        boundaries = boundaries.view(1, -1).expand(heads, -1)
    if boundaries.shape != (heads, (1 << 2) - 1):
        raise ValueError(
            f"boundaries must be [H,{(1 << 2) - 1}], got {tuple(boundaries.shape)}"
        )
    boundaries = boundaries.to(values.device).contiguous()
    flat_norms = norms.contiguous().reshape(-1)
    vectors = contiguous.numel() // dim
    width = math.ceil(dim * 2 / 8)
    packed = torch.empty(vectors, width, dtype=torch.uint8, device=values.device)
    norm16 = torch.empty(vectors, dtype=torch.float16, device=values.device)
    block_d = triton.next_power_of_2(dim)
    _lm_pack_kernel[vectors,](
        contiguous,
        flat_norms,
        boundaries,
        packed,
        norm16,
        value_stride=dim,
        index_stride=width,
        TOKENS=tokens,
        HEADS=heads,
        D=dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    shape = (*values.shape[:-1], width)
    return (packed.reshape(shape), norm16.reshape(values.shape[:-1]))
