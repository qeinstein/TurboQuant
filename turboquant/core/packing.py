import math
import numpy as np
import torch


def pack(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a 1-D int64 index tensor into a uint8 byte tensor.

    Works for any bit-width 1–8 via a flat bit array (numpy packbits).
    Output length is ceil(len(indices) * bits / 8) bytes.
    """
    if not (1 <= bits <= 8):
        raise ValueError(f"bits must be in [1, 8]; got {bits}")
    n = len(indices)
    # Build bit array (MSB first per index), shape (n * bits,)
    bit_cols = [(indices >> (bits - 1 - b)) & 1 for b in range(bits)]
    bits_arr = torch.stack(bit_cols, dim=1).reshape(-1).numpy().astype(np.uint8)
    # Pad to multiple of 8 then packbits
    pad = (-len(bits_arr)) % 8
    if pad:
        bits_arr = np.pad(bits_arr, (0, pad))
    return torch.from_numpy(np.packbits(bits_arr))


def unpack(packed: torch.Tensor, bits: int, length: int) -> torch.Tensor:
    """Unpack a uint8 byte tensor back into int64 indices.

    Args:
        packed: output of pack().
        bits: bit-width used during packing.
        length: number of indices to recover (original d).
    """
    if not (1 <= bits <= 8):
        raise ValueError(f"bits must be in [1, 8]; got {bits}")
    raw = np.unpackbits(packed.numpy())  # flat bit array, MSB first
    # Take only the bits we need (discard padding)
    raw = raw[: length * bits]
    # Reshape to (length, bits) then decode each row as a binary number
    rows = raw.reshape(length, bits).astype(np.int64)
    powers = 1 << np.arange(bits - 1, -1, -1, dtype=np.int64)  # [2^(b-1), ..., 1]
    return torch.from_numpy((rows * powers).sum(axis=1))
