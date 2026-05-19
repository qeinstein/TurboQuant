import numpy as np
import pytest
from turboquant.core.lloyd_max import lloyd_max, mse_cost, vector_mse_cost, _coord_pdf


D = 128  # representative dimension for codebook tests


def test_pdf_integrates_to_one():
    from scipy import integrate
    val, _ = integrate.quad(lambda x: _coord_pdf(x, D), -1, 1)
    assert abs(val - 1.0) < 1e-5


def test_pdf_is_symmetric():
    xs = np.linspace(-0.99, 0.99, 200)
    assert np.allclose(_coord_pdf(xs, D), _coord_pdf(-xs, D), atol=1e-10)


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_centroids_sorted_and_in_range(bits):
    c = lloyd_max(D, n_levels=2**bits)
    assert np.all(np.diff(c) > 0), "centroids not sorted"
    assert c[0] >= -1.0 and c[-1] <= 1.0


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_centroids_symmetric(bits):
    c = lloyd_max(D, n_levels=2**bits)
    assert np.allclose(c, -c[::-1], atol=1e-4), "centroids not symmetric around 0"


# Paper Theorem 1 gives D_mse ≈ 0.36, 0.117, 0.03, 0.009 for b=1,2,3,4.
# We allow 20% tolerance since d=128 is finite (paper states asymptotic values).
@pytest.mark.parametrize("bits,expected", [(1, 0.36), (2, 0.117), (3, 0.03), (4, 0.009)])
def test_vector_mse_matches_paper(bits, expected):
    # Paper Theorem 1 reports vector-level MSE = d × per-coord MSE.
    c = lloyd_max(D, n_levels=2**bits)
    cost = vector_mse_cost(c, D)
    assert cost < expected * 1.20, f"b={bits}: vector MSE {cost:.4f} exceeds paper bound {expected}"
    assert cost > expected * 0.10, f"b={bits}: vector MSE {cost:.4f} suspiciously low"
