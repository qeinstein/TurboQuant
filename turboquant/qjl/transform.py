import torch


def make_projection(d: int, m: int, seed: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Draw S ∈ R^(m×d) with i.i.d. N(0,1) entries (Definition 3.1, QJL paper arXiv:2406.03482)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(m, d, generator=gen, device=device)


def qjl_encode(S: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress key vector k into a 1-bit QJL sketch.

    QJL paper Algorithm 1 / Definition 3.1:
      k̃ = sign(S·k),   ν = ‖k‖₂

    Args:
        S: (m, d) projection matrix.
        k: (d,) or (n, d) key vector(s).

    Returns:
        k_tilde: same leading dims as k but last dim m, dtype int8, values ±1.
        nu: scalar (or (n,)) L2 norm of each original key.
    """
    projected = k @ S.T  # (..., m)
    k_tilde = projected.sign().to(torch.int8)
    nu = torch.linalg.norm(k.float(), dim=-1)
    return k_tilde, nu
