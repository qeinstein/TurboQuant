import torch


def make_rotation(d: int, seed: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Return a Haar-distributed random orthogonal matrix Π of shape (d, d).

    Uses QR decomposition of a standard Gaussian matrix. The Q factor is
    uniformly distributed over O(d) when the diagonal of R is made positive
    (the 'qr' mode in torch already handles this correctly for Haar measure
    via the sign convention).

    Args:
        d: Vector dimension.
        seed: RNG seed — use a deterministic function of (layer_idx, head_idx)
              so the same rotation is reproduced at dequantization time without
              storing the matrix.
        device: Target device.

    Returns:
        Π: (d, d) orthogonal tensor, dtype float32.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen, device=device)
    Q, R = torch.linalg.qr(G)
    # Flip columns so the diagonal of R is positive → unique Haar sample.
    signs = torch.sign(torch.diag(R))
    Q = Q * signs.unsqueeze(0)
    return Q


def rotation_seed(layer_idx: int, head_idx: int) -> int:
    """Deterministic seed from (layer, head) indices.

    Cantor pairing function — collision-free for non-negative integers.
    """
    return (layer_idx + head_idx) * (layer_idx + head_idx + 1) // 2 + head_idx
