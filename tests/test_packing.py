import torch
import pytest
from turboquant.core.packing import pack, unpack


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("length", [64, 128, 130])
def test_roundtrip(bits, length):
    indices = torch.randint(0, 2**bits, (length,), dtype=torch.int64)
    packed = pack(indices, bits)
    recovered = unpack(packed, bits, length)
    assert torch.equal(indices, recovered)


@pytest.mark.parametrize("bits,length,expected_bytes", [
    (1, 8, 1), (2, 8, 2), (4, 8, 4), (8, 8, 8),
    (4, 128, 64), (2, 128, 32), (1, 128, 16),
])
def test_packed_size(bits, length, expected_bytes):
    indices = torch.zeros(length, dtype=torch.int64)
    assert len(pack(indices, bits)) == expected_bytes


def test_unsupported_bits_raises():
    with pytest.raises(ValueError):
        pack(torch.zeros(8, dtype=torch.int64), bits=9)
