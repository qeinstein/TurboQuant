"""Latency benchmark for quantize/dequantize and inner product estimation.

Measures time per token for each operation at various bit-widths and dimensions.
Run with: python benchmarks/speed_benchmark.py
"""

import time
import torch
from turboquant import TurboQuantMSE, TurboQuantProd

DIMS = [64, 128, 256]
BITS = [2, 4]
N_WARMUP = 50
N_ITERS = 500


def bench(fn, n_warmup=N_WARMUP, n_iters=N_ITERS) -> float:
    for _ in range(n_warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1e6  # µs per call


def run():
    torch.manual_seed(0)

    print(f"{'op':<22} {'dim':>5} {'bits':>4} {'µs/token':>10}")
    print("-" * 46)

    for d in DIMS:
        for bits in BITS:
            mse_q = TurboQuantMSE(d, bits, layer_idx=1, head_idx=0)
            prod_q = TurboQuantProd(d, bits, layer_idx=1, head_idx=1)

            x = torch.randn(d)
            q = torch.randn(d)
            mse_state = mse_q.quantize(x)
            prod_state = prod_q.quantize(x)

            t = bench(lambda: mse_q.quantize(x))
            print(f"{'MSE quantize':<22} {d:>5} {bits:>4} {t:>10.2f}")

            t = bench(lambda: mse_q.dequantize(mse_state))
            print(f"{'MSE dequantize':<22} {d:>5} {bits:>4} {t:>10.2f}")

            t = bench(lambda: prod_q.quantize(x))
            print(f"{'Prod quantize':<22} {d:>5} {bits:>4} {t:>10.2f}")

            t = bench(lambda: prod_q.inner_product(q, prod_state))
            print(f"{'Prod inner_product':<22} {d:>5} {bits:>4} {t:>10.2f}")

        print()


if __name__ == "__main__":
    run()
