import torch


def pack(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a 1-D int64 index tensor into a uint8 byte tensor.

    Indices must be in [0, 2**bits). Bits must divide evenly into 8 or
    be packed with zero-padding on the last byte if len(indices) is not
    a multiple of (8 // bits).

    Args:
        indices: (d,) int64 values in [0, 2**bits).
        bits: bit-width per index (1, 2, 4, or 8).

    Returns:
        packed: (ceil(d * bits / 8),) uint8 tensor.
    """
    if bits not in (1, 2, 4, 8):
        raise ValueError(f"bits must be 1, 2, 4, or 8; got {bits}")
    if bits == 8:
        return indices.to(torch.uint8)

    per_byte = 8 // bits
    # Pad to a multiple of per_byte with zeros.
    pad = (-len(indices)) % per_byte
    if pad:
        indices = torch.cat([indices, indices.new_zeros(pad)])

    packed = torch.zeros(len(indices) // per_byte, dtype=torch.uint8)
    for slot in range(per_byte):
        shift = (per_byte - 1 - slot) * bits
        packed |= (indices[slot::per_byte].to(torch.uint8) << shift)
    return packed


def unpack(packed: torch.Tensor, bits: int, length: int) -> torch.Tensor:
    """Unpack a uint8 byte tensor back into int64 indices.

    Args:
        packed: (ceil(length * bits / 8),) uint8 tensor.
        bits: bit-width used during packing.
        length: number of indices to recover (original d).

    Returns:
        indices: (length,) int64 tensor.
    """
    if bits not in (1, 2, 4, 8):
        raise ValueError(f"bits must be 1, 2, 4, or 8; got {bits}")
    if bits == 8:
        return packed[:length].to(torch.int64)

    per_byte = 8 // bits
    mask = (1 << bits) - 1
    out = torch.zeros(len(packed) * per_byte, dtype=torch.int64)
    for slot in range(per_byte):
        shift = (per_byte - 1 - slot) * bits
        out[slot::per_byte] = (packed.to(torch.int64) >> shift) & mask
    return out[:length]
