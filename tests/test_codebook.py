import torch
import pytest
from turboquant.core.codebook import get_codebook, quantize_coords, dequantize_coords


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_codebook_shape_and_range(bits):
    cb = get_codebook(128, bits)
    assert cb.shape == (2**bits,)
    assert cb[0] >= -1.0 and cb[-1] <= 1.0


def test_codebook_is_cached():
    cb1 = get_codebook(128, 4)
    cb2 = get_codebook(128, 4)
    assert cb1.data_ptr() == cb2.data_ptr()


def test_quantize_nearest_centroid():
    cb = get_codebook(128, 2)
    y = cb.clone()  # exact centroid values
    idx = quantize_coords(y.unsqueeze(0), cb).squeeze(0)
    assert torch.equal(idx, torch.arange(4))


def test_roundtrip_recovers_centroid():
    cb = get_codebook(128, 3)
    y = torch.randn(64)
    idx = quantize_coords(y, cb)
    y_hat = dequantize_coords(idx, cb)
    # Every reconstructed value must be one of the codebook centroids.
    for val in y_hat:
        assert any(torch.isclose(val, c) for c in cb)


def test_unsupported_bits_raises():
    with pytest.raises(ValueError):
        get_codebook(128, 5)
