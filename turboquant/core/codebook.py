import numpy as np
import torch
from functools import lru_cache
from turboquant.core.lloyd_max import lloyd_max

SUPPORTED_BITS = (1, 2, 3, 4)


@lru_cache(maxsize=64)
def get_codebook(d: int, bits: int) -> torch.Tensor:
    """Return cached Lloyd-Max centroids for dimension d and bit-width bits.

    Centroids are computed once per (d, bits) pair and cached in memory.
    Shape: (2**bits,), dtype float32, values in [-1, 1], sorted ascending.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    centroids = lloyd_max(d, n_levels=2**bits)
    return torch.tensor(centroids, dtype=torch.float32)


def quantize_coords(y: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Map each coordinate in y to the index of the nearest centroid.

    Args:
        y: (..., d) rotated coordinates, values in [-1, 1].
        codebook: (K,) sorted centroids.

    Returns:
        indices: (..., d) int64 indices into codebook.
    """
    # Euclidean distance to each centroid; argmin over centroid axis.
    dists = (y.unsqueeze(-1) - codebook) ** 2  # (..., d, K)
    return dists.argmin(dim=-1)


def dequantize_coords(indices: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Retrieve centroid values for each index.

    Args:
        indices: (..., d) int64 indices.
        codebook: (K,) centroids.

    Returns:
        (..., d) float32 reconstructed coordinates.
    """
    return codebook[indices]
