"""Compression ratio benchmark — no model required.

Reports bytes per token for keys and values at various bit-widths,
compared to fp16 baseline.
"""

import torch
from turboquant import TurboQuantMSE, TurboQuantProd

DIMS = [64, 128, 256]
KEY_BITS = [2, 3, 4]
VAL_BITS = [2, 3, 4]


def key_compressed_bytes(d: int, bits: int) -> int:
    q = TurboQuantProd(d, bits, layer_idx=1, head_idx=0)
    k = torch.randn(d)
    s = q.quantize(k)
    return len(s["packed"]) + len(s["r_tilde"]) // 8 + 1 + 4 + 4
    # packed MSE indices + QJL bits (d bits = d/8 bytes) + ceil remainder + 2 fp32 norms


def val_compressed_bytes(d: int, bits: int) -> int:
    q = TurboQuantMSE(d, bits, layer_idx=1, head_idx=0)
    v = torch.randn(d)
    s = q.quantize(v)
    return len(s["packed"]) + 4  # packed indices + fp32 norm


def fp16_bytes(d: int) -> int:
    return d * 2


def run():
    print(f"{'dim':>6}  {'bits':>4}  {'key fp16':>8}  {'key comp':>8}  {'key ratio':>9}  "
          f"{'val fp16':>8}  {'val comp':>8}  {'val ratio':>9}")
    print("-" * 78)

    for d in DIMS:
        for kb, vb in zip(KEY_BITS, VAL_BITS):
            kc = key_compressed_bytes(d, kb)
            vc = val_compressed_bytes(d, vb)
            kf = fp16_bytes(d)
            vf = fp16_bytes(d)
            print(f"{d:>6}  {kb}/{vb}  {kf:>8}  {kc:>8}  {kf/kc:>8.1f}x  "
                  f"{vf:>8}  {vc:>8}  {vf/vc:>8.1f}x")

    print()
    print("Combined KV compression (key 4-bit + val 2-bit vs fp16 key+val):")
    for d in DIMS:
        kc = key_compressed_bytes(d, 4)
        vc = val_compressed_bytes(d, 2)
        total_fp16 = fp16_bytes(d) * 2
        total_comp = kc + vc
        print(f"  d={d}: {total_fp16}B → {total_comp}B  ({total_fp16/total_comp:.1f}x)")


if __name__ == "__main__":
    run()
