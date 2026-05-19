import torch
import pytest
from turboquant.quantizer import TurboQuantMSE, TurboQuantProd

D = 128
N_TRIALS = 5_000


# ── TurboQuantMSE ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_mse_roundtrip_shape(bits):
    q = TurboQuantMSE(D, bits)
    x = torch.randn(D)
    state = q.quantize(x)
    x_hat = q.dequantize(state)
    assert x_hat.shape == (D,)


@pytest.mark.parametrize("bits,threshold", [(4, 0.015), (3, 0.05), (2, 0.15), (1, 0.45)])
def test_mse_distortion_within_paper_bound(bits, threshold):
    # Paper Theorem 1: D_mse <= threshold for these bit-widths at d=128.
    q = TurboQuantMSE(D, bits)
    mse_vals = []
    torch.manual_seed(99)  # must differ from rotation_seed(0,0)=0
    for _ in range(500):
        x = torch.randn(D)
        x = x / x.norm()
        x_hat = q.dequantize(q.quantize(x))
        mse_vals.append(((x - x_hat) ** 2).sum().item())
    avg_mse = sum(mse_vals) / len(mse_vals)
    assert avg_mse < threshold, f"b={bits}: avg MSE {avg_mse:.4f} > {threshold}"


def test_mse_norm_preserved_after_roundtrip():
    q = TurboQuantMSE(D, bits=4)
    x = torch.randn(D) * 5.0
    x_hat = q.dequantize(q.quantize(x))
    ratio = x_hat.norm() / x.norm()
    assert 0.9 < ratio.item() < 1.1


# ── TurboQuantProd ─────────────────────────────────────────────────────────────

def test_prod_requires_bits_ge_2():
    with pytest.raises(ValueError):
        TurboQuantProd(D, bits=1)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_prod_roundtrip_shape(bits):
    q = TurboQuantProd(D, bits)
    x = torch.randn(D)
    state = q.quantize(x)
    x_hat = q.dequantize(state)
    assert x_hat.shape == (D,)


def test_inner_product_unbiased():
    # E[<q, x_hat>] = <q, x>  (TurboQuant paper, Theorem 2 + unbiasedness condition)
    torch.manual_seed(42)
    x = torch.randn(D)
    x = x / x.norm()
    query = torch.randn(D)
    true_ip = torch.dot(query, x).item()

    estimates = []
    for seed in range(N_TRIALS):
        q = TurboQuantProd(D, bits=4, layer_idx=seed, head_idx=0)
        state = q.quantize(x)
        est = q.inner_product(query, state).item()
        estimates.append(est)

    mean_est = sum(estimates) / N_TRIALS
    assert abs(mean_est - true_ip) < 0.02, (
        f"E[ip_estimate]={mean_est:.4f} != true_ip={true_ip:.4f}"
    )


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_inner_product_matches_dequantize(bits):
    # inner_product() and dot(q, dequantize()) must agree (both use same formula).
    q = TurboQuantProd(D, bits)
    x = torch.randn(D)
    query = torch.randn(D)
    state = q.quantize(x)
    ip_direct = q.inner_product(query, state)
    ip_deq = torch.dot(query, q.dequantize(state))
    assert torch.isclose(ip_direct, ip_deq, atol=1e-4), (
        f"direct={ip_direct:.4f}, via dequant={ip_deq:.4f}"
    )
