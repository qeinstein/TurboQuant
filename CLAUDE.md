# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TurboQuant is a Python implementation of the TurboQuant algorithm — a near-optimal vector quantization method for compressing LLM KV caches while preserving efficient inner product computation. Based on arXiv:2504.19874 (TurboQuant), arXiv:2406.03482 (QJL), and arXiv:2502.02617 (PolarQuant).

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"   # setup
.venv/bin/python -m pytest tests/                           # run all tests
.venv/bin/python -m pytest tests/test_quantizer.py -v      # single file
.venv/bin/python examples/basic_usage.py                   # smoke test
.venv/bin/python benchmarks/memory_benchmark.py            # compression ratios
.venv/bin/python benchmarks/speed_benchmark.py             # latency per token
.venv/bin/python benchmarks/quality_benchmark.py           # perplexity (downloads GPT-2)
ruff check . && ruff format .                               # lint + format
```

Benchmarks requiring HuggingFace: `pip install -e ".[bench]"` first.

## Architecture

Two-stage pipeline:

**Stage 1 — TurboQuantMSE** (`quantizer.py`): MSE-optimal quantization.
1. Normalise input vector, store norm.
2. Rotate: `y = Π·x` using a Haar-distributed random orthogonal matrix (`core/rotation.py`). After rotation, each coordinate follows `fX(x) = C·(1-x²)^((d-3)/2)` on [-1,1] (Lemma 1, TurboQuant paper).
3. Quantize each coordinate to the nearest Lloyd-Max centroid (`core/lloyd_max.py`, `core/codebook.py`). Codebooks are precomputed once per `(d, bits)` and cached.
4. Pack indices into bytes (`core/packing.py`, supports 1–8 bit widths).

**Stage 2 — TurboQuantProd** (`quantizer.py`): corrects inner product bias.
- Runs TurboQuantMSE at `(bits-1)` bit-width.
- Computes residual `r = x - dequantize(quantize(x))`.
- Applies 1-bit QJL to residual: `sign(S·r)`, stores `(sign(S·r), ‖r‖₂)`.
- Inner product estimator: `⟨q, x̂⟩ = ⟨q, x̂_mse⟩ + √(π/2)/d · ‖r‖₂ · ⟨Sq, sign(Sr)⟩`

**KV cache** (`kv_cache.py`): keys use TurboQuantProd (inner-product accuracy needed for attention scores); values use TurboQuantMSE (MSE suffices since they're multiplied by softmax weights).

## Key mathematical facts

- Coordinate distribution after rotation is **not** Beta(d/2, 1/2) — it is `fX(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)`, symmetric on [-1,1].
- `vector_mse_cost = d × per_coord_mse` (paper Theorem 1 values are per-vector).
- Test random seeds must not equal `rotation_seed(layer_idx, head_idx)` — the rotation matrix generator and the test data generator both use the same PyTorch RNG algorithm, so equal seeds produce correlated vectors that inflate MSE measurements.

## Domain context

- **KV cache compression**: stores key/value embeddings from all previous tokens; size scales with context length and is the dominant memory bottleneck at long context.
- **Inner product preservation**: attention scores `softmax([⟨q,k₁⟩,...,⟨q,kₙ⟩])` amplify small errors — compression must keep dot products accurate, not just minimize L2 error.
- **QJL** (arXiv:2406.03482): 1-bit Johnson-Lindenstrauss sketch. Asymmetric estimator `ProdQJL(q,k) = √(π/2)/m · ‖k‖₂ · ⟨Sq, sign(Sk)⟩` is provably unbiased. Applying sign to both vectors estimates angle only — biased.
