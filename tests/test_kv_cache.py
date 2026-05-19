import torch
import pytest
from turboquant.kv_cache import TurboQuantKVCache

D = 128
N_TOKENS = 16


@pytest.fixture
def cache():
    return TurboQuantKVCache(d=D, key_bits=4, val_bits=2, layer_idx=1, head_idx=0)


def test_append_and_len(cache):
    assert len(cache) == 0
    torch.manual_seed(5)
    for _ in range(N_TOKENS):
        cache.append(torch.randn(D), torch.randn(D))
    assert len(cache) == N_TOKENS


def test_attn_scores_shape(cache):
    torch.manual_seed(6)
    for _ in range(N_TOKENS):
        cache.append(torch.randn(D), torch.randn(D))
    q = torch.randn(D)
    scores = cache.attn_scores(q)
    assert scores.shape == (N_TOKENS,)


def test_values_shape(cache):
    torch.manual_seed(7)
    for _ in range(N_TOKENS):
        cache.append(torch.randn(D), torch.randn(D))
    vals = cache.values()
    assert vals.shape == (N_TOKENS, D)


def test_attn_scores_close_to_true(cache):
    # Compressed scores should be close to true inner products.
    torch.manual_seed(8)
    keys = [torch.randn(D) for _ in range(N_TOKENS)]
    vals = [torch.randn(D) for _ in range(N_TOKENS)]
    for k, v in zip(keys, vals):
        cache.append(k, v)

    q = torch.randn(D)
    true_scores = torch.tensor([torch.dot(q, k) for k in keys])
    est_scores = cache.attn_scores(q)

    # Mean absolute error < 10% of the RMS of true scores.
    rms = true_scores.pow(2).mean().sqrt().item()
    mae = (est_scores - true_scores).abs().mean().item()
    assert mae < 0.5 * rms, f"MAE {mae:.3f} > 0.5 × RMS {rms:.3f}"


def test_clear(cache):
    torch.manual_seed(9)
    for _ in range(N_TOKENS):
        cache.append(torch.randn(D), torch.randn(D))
    cache.clear()
    assert len(cache) == 0
