# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from functools import lru_cache
import torch


@lru_cache(maxsize=16)
def hadamard_cpu(dim: int) -> torch.Tensor:
    if dim <= 0 or dim & dim - 1:
        raise ValueError(f"TurboQuant requires power-of-two head_dim, got {dim}")
    h = torch.tensor([[1.0]], dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(dim)).contiguous()


def hadamard(dim: int, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return hadamard_cpu(dim).to(device=device, dtype=dtype)


def _gaussian_pdf(x: float, variance: float) -> float:
    return 1.0 / math.sqrt(2 * math.pi * variance) * math.exp(-x * x / (2 * variance))


def _trapz(fn, start: float, end: float, n: int = 200) -> float:
    step = (end - start) / n
    result = 0.5 * (fn(start) + fn(end))
    for index in range(1, n):
        result += fn(start + index * step)
    return result * step


@lru_cache(maxsize=32)
def lloyd_max(
    dim: int, bits: int = 2, max_iter: int = 200, tol: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    if bits != 2:
        raise ValueError("Only K2/V2 is supported")
    if bits < 1 or bits > 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")
    levels = 2**bits
    variance = 1.0 / dim
    sigma = math.sqrt(variance)

    def pdf(value: float) -> float:
        return _gaussian_pdf(value, variance)

    lo, hi = (-3.5 * sigma, 3.5 * sigma)
    centroids = [lo + (hi - lo) * (i + 0.5) / levels for i in range(levels)]
    for _ in range(max_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(levels - 1)
        ]
        edges = [lo * 3] + boundaries + [hi * 3]
        updated = []
        for index in range(levels):
            start, end = (edges[index], edges[index + 1])
            numerator = _trapz(lambda x: x * pdf(x), start, end)
            denominator = _trapz(pdf, start, end)
            updated.append(
                numerator / denominator if denominator > 1e-15 else centroids[index]
            )
        if max((abs(updated[i] - centroids[i]) for i in range(levels))) < tol:
            break
        centroids = updated
    centroid_tensor = torch.tensor(centroids, dtype=torch.float32)
    boundaries = (centroid_tensor[:-1] + centroid_tensor[1:]) / 2
    return (centroid_tensor, boundaries)
