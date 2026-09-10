import math
from dataclasses import dataclass
import torch
from .turboquant import lloyd_max

CODEC_LM = "lm"
CODEC_LOCAL_LM = "local_lm"
MXFP4_BLOCK_SIZE = 32
GROUP_SIZE = 32


def _encode_mxfp4(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = int(values.shape[-1])
    if width % MXFP4_BLOCK_SIZE or width % 2:
        raise ValueError(
            f"MXFP4 metadata width must be divisible by {MXFP4_BLOCK_SIZE}, got {width}"
        )
    source = values.float()
    blocks = source.reshape(*source.shape[:-1], -1, MXFP4_BLOCK_SIZE)
    maximum = blocks.abs().amax(dim=-1, keepdim=True)
    exponent = torch.ceil(torch.log2((maximum / 6.0).clamp_min(2.0 ** (-127)))).clamp_(
        -127, 128
    )
    exponent = torch.where(maximum == 0, torch.zeros_like(exponent), exponent)
    scale = torch.pow(2.0, exponent)
    normalized = blocks / scale
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=values.device,
    )
    magnitude_code = (normalized.abs().unsqueeze(-1) - magnitudes).abs().argmin(dim=-1)
    code = magnitude_code.to(torch.uint8)
    code = code | (normalized < 0).to(torch.uint8) << 3
    flat_code = code.reshape(*source.shape[:-1], width)
    packed = flat_code[..., 0::2] | flat_code[..., 1::2] << 4
    exponent_byte = (exponent.squeeze(-1).to(torch.int16) + 127).to(torch.uint8)
    reconstructed = (
        magnitudes[magnitude_code] * torch.where(normalized < 0, -1.0, 1.0) * scale
    ).reshape_as(source)
    return (packed.contiguous(), exponent_byte.contiguous(), reconstructed)


def _decode_mxfp4(
    packed: torch.Tensor, exponent_byte: torch.Tensor, width: int
) -> torch.Tensor:
    if packed.shape[-1] != width // 2:
        raise ValueError("MXFP4 packed metadata has the wrong width")
    codes = torch.empty(
        (*packed.shape[:-1], width), dtype=torch.uint8, device=packed.device
    )
    codes[..., 0::2] = packed & 15
    codes[..., 1::2] = packed >> 4
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    values = magnitudes[(codes & 7).long()]
    values = values * torch.where(codes & 8 != 0, -1.0, 1.0)
    scale = torch.pow(
        2.0, exponent_byte.to(torch.int16).float() - 127.0
    ).repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    return values * scale


def pack_unsigned(values):
    if values.shape[-1] % 4:
        raise ValueError("Head dimension must be divisible by four")
    values = values.to(torch.uint8)
    if values.numel() and int(values.max()) > 3:
        raise ValueError("Indices must be between 0 and 3")
    return (
        values[..., 0::4]
        | values[..., 1::4] << 2
        | values[..., 2::4] << 4
        | values[..., 3::4] << 6
    )


def unpack_unsigned(packed):
    shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
    return (packed.unsqueeze(-1) >> shifts & 3).flatten(-2)


@dataclass
class PackedGroup:
    mod: int
    codec: str
    batch: int
    heads: int
    dim: int
    device: torch.device
    capacity: int = 0
    length: int = 0

    def __post_init__(self):
        if self.codec not in (CODEC_LOCAL_LM, CODEC_LM):
            raise ValueError(f"Unsupported codec: {self.codec}")
        if self.dim % 32 or self.dim & self.dim - 1:
            raise ValueError(
                "Head dimension must be a power of two and divisible by 32"
            )
        self.index = torch.empty(0, dtype=torch.uint8, device=self.device)
        self.norm = torch.empty(0, dtype=torch.float16, device=self.device)
        self.positions = torch.empty(0, dtype=torch.int32, device=self.device)
        self.local_min = torch.empty(0, dtype=torch.uint8, device=self.device)
        self.local_span = torch.empty_like(self.local_min)
        self.local_min_scale = torch.empty_like(self.local_min)
        self.local_span_scale = torch.empty_like(self.local_min)
        self.local_pending = torch.empty(0, dtype=torch.float16, device=self.device)
        if self.capacity:
            self._allocate(self.capacity)

    @property
    def index_width(self):
        return self.dim // 4

    def _allocate(self, capacity):
        old = self.length
        for name, shape, dtype, live in [
            (
                "index",
                (self.batch, self.heads, capacity, self.index_width),
                torch.uint8,
                old,
            ),
            ("norm", (self.batch, self.heads, capacity), torch.float16, old),
        ]:
            new = torch.empty(shape, dtype=dtype, device=self.device)
            if old:
                new[:, :, :old].copy_(getattr(self, name)[:, :, :old])
            setattr(self, name, new)
        positions = torch.empty(capacity, dtype=torch.int32, device=self.device)
        positions[:old].copy_(self.positions[:old])
        self.positions = positions
        if self.codec == CODEC_LOCAL_LM:
            groups, old_groups = (
                math.ceil(capacity / GROUP_SIZE),
                math.ceil(old / GROUP_SIZE),
            )
            for name, width in [
                ("local_min", self.dim // 2),
                ("local_span", self.dim // 2),
                ("local_min_scale", self.dim // 32),
                ("local_span_scale", self.dim // 32),
            ]:
                new = torch.empty(
                    self.batch,
                    self.heads,
                    groups,
                    width,
                    dtype=torch.uint8,
                    device=self.device,
                )
                if old:
                    new[:, :, :old_groups].copy_(getattr(self, name)[:, :, :old_groups])
                setattr(self, name, new)
        self.capacity = capacity

    def _reserve(self, needed):
        if needed > self.capacity:
            capacity = needed if self.capacity == 0 else max(needed, 2 * self.capacity)
            self._allocate(capacity)

    def append(self, index, norm, positions):
        n = positions.numel()
        self._reserve(self.length + n)
        target = slice(self.length, self.length + n)
        self.index[:, :, target].copy_(index)
        self.norm[:, :, target].copy_(norm)
        self.positions[target].copy_(positions.to(self.device, dtype=torch.int32))
        self.length += n

    def append_local_lm(
        self, values: torch.Tensor, norms: torch.Tensor, positions: torch.Tensor
    ) -> None:
        if self.codec != CODEC_LOCAL_LM:
            raise ValueError("append_local_lm requires the local_lm codec")
        if values.ndim != 4 or values.shape[:2] != (self.batch, self.heads):
            raise ValueError(f"invalid local K shape: {tuple(values.shape)}")
        if values.shape[-1] != self.dim or norms.shape != values.shape[:-1]:
            raise ValueError("local K values/norms have incompatible shapes")
        n = int(values.shape[2])
        if n != int(positions.numel()):
            raise ValueError("local K positions do not match token count")
        if n == 0:
            return
        tail = self.length % 32
        prefix = self.length - tail
        source = values.to(torch.float16)
        new_norms = norms.to(torch.float16)
        new_positions = positions.to(device=self.device, dtype=torch.int32)
        if tail:
            if self.local_pending.shape != (self.batch, self.heads, tail, self.dim):
                raise RuntimeError("local K pending group is inconsistent")
            source = torch.cat((self.local_pending, source), dim=2)
            new_norms = torch.cat(
                (self.norm[:, :, prefix : self.length].clone(), new_norms), dim=2
            )
            new_positions = torch.cat(
                (self.positions[prefix : self.length].clone(), new_positions), dim=0
            )
        updated = int(source.shape[2])
        self._reserve(prefix + updated)
        centroids = lloyd_max(self.dim, 2)[0].to(self.device)
        levels = (centroids - centroids[0]) / (centroids[-1] - centroids[0])
        boundaries = (levels[:-1] + levels[1:]) / 2
        metadata_minimum = None
        metadata_span = None
        groups = math.ceil(updated / 32)
        padded_tokens = groups * 32
        padded = torch.zeros(
            self.batch,
            self.heads,
            padded_tokens,
            self.dim,
            dtype=source.dtype,
            device=self.device,
        )
        padded[:, :, :updated].copy_(source)
        grouped = padded.reshape(self.batch, self.heads, groups, 32, self.dim).float()
        valid = (torch.arange(padded_tokens, device=self.device) < updated).reshape(
            groups, 32
        )
        valid = valid.view(1, 1, groups, 32, 1)
        minimum = torch.where(
            valid, grouped, torch.full_like(grouped, float("inf"))
        ).amin(dim=3)
        maximum = torch.where(
            valid, grouped, torch.full_like(grouped, -float("inf"))
        ).amax(dim=3)
        raw_span = (maximum - minimum).clamp_min_(1e-12)
        stored_min, min_scale, metadata_minimum = _encode_mxfp4(minimum)
        stored_span, span_scale, metadata_span = _encode_mxfp4(raw_span)
        group_slice = slice(prefix // 32, prefix // 32 + groups)
        self.local_min[:, :, group_slice].copy_(stored_min)
        self.local_span[:, :, group_slice].copy_(stored_span)
        self.local_min_scale[:, :, group_slice].copy_(min_scale)
        self.local_span_scale[:, :, group_slice].copy_(span_scale)
        metadata_minimum = metadata_minimum.to(torch.float16).contiguous()
        metadata_span = metadata_span.to(torch.float16).contiguous()
        if source.is_cuda:
            from .triton_store import pack_local_lm_into

            pack_local_lm_into(
                source,
                new_norms,
                boundaries,
                self.index,
                self.norm,
                token_offset=prefix,
                metadata_minimum=metadata_minimum,
                metadata_span=metadata_span,
            )
        else:
            packed_chunks = []
            for start in range(0, updated, 32):
                stop = min(start + 32, updated)
                chunk = source[:, :, start:stop].float()
                minimum = chunk.amin(dim=2)
                maximum = chunk.amax(dim=2)
                span = (maximum - minimum).clamp_min_(1e-12)
                stored_min, min_scale, reconstructed_minimum = _encode_mxfp4(minimum)
                stored_span, span_scale, reconstructed_span = _encode_mxfp4(span)
                minimum = reconstructed_minimum
                span = reconstructed_span
                safe_span = torch.where(span == 0, torch.ones_like(span), span)
                unit = ((chunk - minimum.unsqueeze(2)) / safe_span.unsqueeze(2)).clamp_(
                    0.0, 1.0
                )
                code = torch.bucketize(unit.contiguous(), boundaries, right=True)
                packed_chunks.append(pack_unsigned(code.to(torch.uint8)))
                group_index = prefix // 32 + start // 32
                self.local_min[:, :, group_index].copy_(stored_min)
                self.local_span[:, :, group_index].copy_(stored_span)
                self.local_min_scale[:, :, group_index].copy_(min_scale)
                self.local_span_scale[:, :, group_index].copy_(span_scale)
            packed = torch.cat(packed_chunks, dim=2)
            self.index[:, :, prefix : prefix + updated].copy_(packed)
            self.norm[:, :, prefix : prefix + updated].copy_(new_norms)
        self.positions[prefix : prefix + updated].copy_(new_positions)
        self.length = prefix + updated
        pending = self.length % 32
        self.local_pending = (
            source[:, :, -pending:].clone()
            if pending
            else torch.empty(0, dtype=torch.float16, device=self.device)
        )

    def unpack_rotated(self):
        n = self.length
        indexes = unpack_unsigned(self.index[:, :, :n]).long()
        centroids = lloyd_max(self.dim)[0].to(self.device)
        if self.codec == CODEC_LOCAL_LM:
            levels = (centroids - centroids[0]) / (centroids[-1] - centroids[0])
            group_ids = torch.arange(n, device=self.device) // GROUP_SIZE
            minimum = _decode_mxfp4(
                self.local_min, self.local_min_scale, self.dim
            ).index_select(2, group_ids)
            span = _decode_mxfp4(
                self.local_span, self.local_span_scale, self.dim
            ).index_select(2, group_ids)
            values = minimum + span * levels[indexes]
        else:
            values = centroids[indexes]
            values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return (
            values * self.norm[:, :, :n].float().unsqueeze(-1),
            self.positions[:n].long(),
        )

    def storage_bytes(self, *, live=False):
        if not live:
            return sum(
                (
                    int(t.untyped_storage().nbytes())
                    for t in (
                        self.index,
                        self.norm,
                        self.positions,
                        self.local_min,
                        self.local_span,
                        self.local_min_scale,
                        self.local_span_scale,
                        self.local_pending,
                    )
                )
            )
        total = (
            self.batch * self.heads * self.length * (self.index_width + 2)
            + self.length * 4
        )
        if self.codec == CODEC_LOCAL_LM:
            total += (
                self.batch
                * self.heads
                * math.ceil(self.length / GROUP_SIZE)
                * (self.dim + 2 * self.dim // 32)
            )
            total += self.local_pending.numel() * self.local_pending.element_size()
        return total


class PackedLayer:
    def __init__(self, batch, heads, dim, dtype, device):
        self.batch, self.heads, self.dim = (batch, heads, dim)
        self.dtype, self.device = (dtype, device)
        self.groups = {}

    def group(self, mod, codec):
        if mod not in self.groups:
            self.groups[mod] = PackedGroup(
                mod, codec, self.batch, self.heads, self.dim, self.device
            )
        group = self.groups[mod]
        if group.codec != codec:
            raise ValueError("A modality cannot change codec within a cache")
        return group

    def storage_bytes(self, *, live=False):
        return sum((group.storage_bytes(live=live) for group in self.groups.values()))
