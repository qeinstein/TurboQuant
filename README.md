# TurboQuant

Python implementation of TurboQuant — a near-optimal vector quantization algorithm for LLM KV cache compression and efficient inner product preservation.

Based on:
- [TurboQuant](https://arxiv.org/abs/2504.19874) (arXiv:2504.19874) — the main algorithm
- [QJL](https://arxiv.org/abs/2406.03482) (arXiv:2406.03482) — the 1-bit inner product corrector
- [PolarQuant](https://arxiv.org/abs/2502.02617) (arXiv:2502.02617) — related preconditioning work

## How it works

Two-stage pipeline per key vector:

1. **MSE stage** — rotate the vector with a random orthogonal matrix, then apply a Lloyd-Max scalar quantizer per coordinate. The rotation induces a known distribution on each coordinate, so the optimal codebook can be precomputed analytically.

2. **Inner product correction** — the MSE quantizer introduces bias in dot products. A 1-bit QJL sketch of the residual corrects this, giving an unbiased inner product estimator.

Values only need Stage 1 (softmax weights absorb the MSE). Keys need both stages (attention scores require accurate inner products).

**Compression** (d=128 head dim, 4-bit keys + 2-bit values): ~4.7x vs fp16.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e .
```

For benchmarks (requires downloading GPT-2 + WikiText-2):

```bash
.venv/bin/pip install -e ".[bench]"
```

## Usage

```python
import torch
from turboquant import TurboQuantKVCache

cache = TurboQuantKVCache(d=128, key_bits=4, val_bits=2, layer_idx=0, head_idx=0)

# Compress and store tokens
for k, v in zip(keys, values):
    cache.append(k, v)

# Attention
scores  = cache.attn_scores(query) / d**0.5   # inner products with all cached keys
weights = torch.softmax(scores, dim=0)
output  = (weights.unsqueeze(1) * cache.values()).sum(0)
```

See `examples/basic_usage.py` for more.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

82 tests covering unbiasedness of the inner product estimator, MSE bounds from the paper, bit-packing roundtrips, and end-to-end cache correctness.

## Benchmarks

```bash
.venv/bin/python benchmarks/memory_benchmark.py   # compression ratios
.venv/bin/python benchmarks/speed_benchmark.py    # latency per token (CPU)
.venv/bin/python benchmarks/quality_benchmark.py  # perplexity vs fp16 on WikiText-2
```

Quick mode (runs in ~3 minutes, 512 tokens, 4-bit only):

```bash
.venv/bin/python benchmarks/quality_benchmark.py --quick
```

## Results (d=128, CPU)

| Config | Compression | Softmax weight error |
|---|---|---|
| key 4-bit + val 2-bit | 4.7x | ~0.006 mean abs |
| key 3-bit + val 2-bit | 3.5x | higher |
| key 2-bit + val 2-bit | 2.5x | higher |

Output cosine similarity vs fp16 at 4-bit keys: **~0.92** on a single attention head with 32 tokens.

Full perplexity numbers: run `quality_benchmark.py`.

## Experiments

The quality benchmark includes three variants on top of the base TurboQuant algorithm. All numbers below are perplexity on WikiText-2 (GPT-2, 512 tokens, 4-bit keys + 2-bit values, fp16 baseline = 31.00).

| Variant | PPL | Δ vs fp16 |
|---|---|---|
| fp16 baseline | 31.00 | — |
| TurboQuant base | 44.78 | +13.78 |
| **TurboQuant m=2d sketch** | **38.58** | **+7.59** |
| TurboQuant learned rotation | 102.08 | +71.08 |
| Polar + QJL | 60.32 | +29.33 |

### Larger QJL sketch (m = 2d)

The original TurboQuant paper fixes the QJL sketch matrix at S ∈ ℝ^(d×d), i.e. m = d, and does not vary this. The QJL paper discusses theoretically that error scales as 1/√m but only uses m = d in experiments. Neither paper nor PolarQuant explores m > d in practice.

Using m = 2d (doubling the sketch size) reduces the perplexity gap from +13.78 to +7.59 — nearly halving the quality loss at the same bit-width. The tradeoff is slightly more memory for the sketch and a larger matrix multiply at inference. This appears to be a novel practical finding not present in any of the three source papers.

### Learned rotation (PCA)

Instead of the random Haar rotation, this calibrates a per-head rotation from actual GPT-2 key vectors using SVD (principal components). Result: perplexity gets much worse (+71.08 vs +13.78). The random rotation is not a heuristic — it is theoretically correct. The Lloyd-Max codebook is derived for the arc-sine coordinate distribution that only holds after a Haar-distributed rotation of a unit vector. A data-driven rotation breaks this assumption and the codebook no longer matches the actual distribution.

### Polar + QJL

Replaces the MSE stage with a polar coordinate transform (following PolarQuant): vectors are recursively decomposed into angles across log₂(d) levels, with level 0 quantized at 4 bits and levels 1+ at 2 bits. QJL correction is then applied to the residual. Result: worse than base TurboQuant (+29.33 vs +13.78). The polar quantization at these bit widths introduces more MSE than the 3-bit Lloyd-Max scalar quantizer, and the QJL correction cannot compensate. Negative result, but confirms that TurboQuant's MSE stage is well-optimised for this regime.
