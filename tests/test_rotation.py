import torch
import pytest
from turboquant.core.rotation import make_rotation, rotation_seed


@pytest.mark.parametrize("d", [32, 64, 128])
def test_orthogonality(d):
    Pi = make_rotation(d, seed=0)
    I = torch.eye(d)
    assert torch.allclose(Pi.T @ Pi, I, atol=1e-5), "Π^T Π ≠ I"
    assert torch.allclose(Pi @ Pi.T, I, atol=1e-5), "Π Π^T ≠ I"


@pytest.mark.parametrize("d", [64, 128])
def test_norm_preservation(d):
    Pi = make_rotation(d, seed=1)
    x = torch.randn(d)
    norm_x = torch.linalg.norm(x)
    norm_rotated = torch.linalg.norm(Pi @ x)
    assert torch.allclose(norm_x, norm_rotated, atol=1e-5)


@pytest.mark.parametrize("d", [64, 128])
def test_inner_product_preservation(d):
    Pi = make_rotation(d, seed=2)
    x = torch.randn(d)
    y = torch.randn(d)
    assert torch.allclose(torch.dot(x, y), torch.dot(Pi @ x, Pi @ y), atol=1e-4)


def test_reproducibility():
    Pi1 = make_rotation(64, seed=42)
    Pi2 = make_rotation(64, seed=42)
    assert torch.allclose(Pi1, Pi2)


def test_different_seeds_differ():
    Pi1 = make_rotation(64, seed=0)
    Pi2 = make_rotation(64, seed=1)
    assert not torch.allclose(Pi1, Pi2)


def test_rotation_seed_no_collisions():
    seen = set()
    for layer in range(32):
        for head in range(32):
            s = rotation_seed(layer, head)
            assert s not in seen, f"Collision at layer={layer}, head={head}"
            seen.add(s)
