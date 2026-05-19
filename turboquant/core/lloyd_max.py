import numpy as np
from scipy import integrate, optimize


def _coord_pdf(x: np.ndarray, d: int) -> np.ndarray:
    # Lemma 1, TurboQuant paper (arXiv:2504.19874):
    # fX(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2), x ∈ [-1, 1]
    # Each coordinate of a Haar-rotated unit vector follows this distribution.
    from scipy.special import gamma
    C = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    vals = np.where(np.abs(x) < 1.0, C * (1.0 - x**2) ** ((d - 3) / 2), 0.0)
    return vals


def lloyd_max(d: int, n_levels: int, tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """Lloyd-Max optimal scalar quantizer for the coordinate distribution of a
    Haar-rotated unit vector (Lemma 1, TurboQuant arXiv:2504.19874, Eq. 4).

    Solves the continuous k-means problem:
      min_{c1..cK} Σ_i ∫_{t_{i-1}}^{t_i} (x - c_i)² fX(x) dx
    where boundaries t_i = (c_i + c_{i+1}) / 2 (Voronoi condition).

    Args:
        d: Vector dimension (determines distribution shape).
        n_levels: Number of quantization levels (2^b for b-bit quantization).
        tol: Convergence threshold on max centroid movement.
        max_iter: Maximum iterations.

    Returns:
        centroids: (n_levels,) sorted array of optimal centroids in [-1, 1].
    """
    # Initialise centroids at distribution quantiles so convergence is fast.
    # For this symmetric distribution, space them symmetrically in (-1, 1).
    eps = 1e-6
    centroids = np.linspace(-1 + eps, 1 - eps, n_levels)

    for _ in range(max_iter):
        # Voronoi boundaries: midpoints between consecutive centroids.
        boundaries = np.concatenate([[-1.0], (centroids[:-1] + centroids[1:]) / 2, [1.0]])

        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]

            # c_i* = E[X | lo ≤ X ≤ hi] = ∫ x·fX(x)dx / ∫ fX(x)dx
            num, _ = integrate.quad(lambda x: x * _coord_pdf(x, d), lo, hi, limit=100)
            den, _ = integrate.quad(lambda x: _coord_pdf(x, d), lo, hi, limit=100)

            # Empty bin: keep the old centroid. Shouldn't happen with good init.
            new_centroids[i] = num / den if den > 1e-15 else centroids[i]

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids


def mse_cost(centroids: np.ndarray, d: int) -> float:
    """Per-coordinate MSE of a codebook against the coordinate distribution (Eq. 4).

    The full vector MSE is d × mse_cost (paper Theorem 1 reports vector-level values).
    """
    boundaries = np.concatenate([[-1.0], (centroids[:-1] + centroids[1:]) / 2, [1.0]])
    cost = 0.0
    for i, c in enumerate(centroids):
        lo, hi = boundaries[i], boundaries[i + 1]
        val, _ = integrate.quad(
            lambda x: (x - c) ** 2 * _coord_pdf(x, d), lo, hi, limit=100
        )
        cost += val
    return cost


def vector_mse_cost(centroids: np.ndarray, d: int) -> float:
    """Full vector MSE = d × per-coordinate MSE. Matches Theorem 1 values."""
    return d * mse_cost(centroids, d)
