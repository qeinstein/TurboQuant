import math
import torch


def prod_qjl(S: torch.Tensor, q: torch.Tensor, k_tilde: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Asymmetric inner product estimator (Equation 4, QJL paper arXiv:2406.03482).

    ProdQJL(q, k) = sqrt(π/2) / m · ‖k‖₂ · ⟨S·q, sign(S·k)⟩

    Query q is projected but NOT sign-quantized. Key k is represented by its
    sign sketch k_tilde = sign(S·k) and norm nu = ‖k‖₂. The asymmetry is
    essential: applying sign to both vectors estimates angle only → biased.

    Args:
        S: (m, d) shared projection matrix.
        q: (d,) current query vector.
        k_tilde: (m,) or (n, m) int8 ±1 sign sketch of cached key(s).
        nu: scalar or (n,) norm of cached key(s).

    Returns:
        estimate: scalar or (n,) inner product estimate(s).
    """
    m = S.shape[0]
    Sq = (S @ q.float())  # (m,)
    # dot(Sq, k_tilde) over the m dimension
    dots = (Sq * k_tilde.float()).sum(dim=-1)  # scalar or (n,)
    return (math.sqrt(math.pi / 2) / m) * nu * dots
