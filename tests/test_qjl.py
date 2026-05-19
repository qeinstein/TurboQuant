import math
import torch
import pytest
from turboquant.qjl.transform import make_projection, qjl_encode
from turboquant.qjl.estimator import prod_qjl

D = 128
M = 128
N_TRIALS = 10_000


def test_k_tilde_is_pm1():
    S = make_projection(D, M, seed=0)
    k = torch.randn(D)
    k_tilde, _ = qjl_encode(S, k)
    assert set(k_tilde.unique().tolist()) <= {-1, 1}


def test_nu_is_l2_norm():
    S = make_projection(D, M, seed=0)
    k = torch.randn(D)
    _, nu = qjl_encode(S, k)
    assert torch.isclose(nu, torch.linalg.norm(k), atol=1e-5)


def test_estimator_unbiased():
    # E[ProdQJL(q,k)] = ⟨q,k⟩ (Lemma 3.2, QJL paper)
    torch.manual_seed(0)
    q = torch.randn(D)
    k = torch.randn(D)
    true_ip = torch.dot(q, k).item()

    estimates = []
    for seed in range(N_TRIALS):
        S = make_projection(D, M, seed=seed)
        k_tilde, nu = qjl_encode(S, k)
        est = prod_qjl(S, q, k_tilde, nu).item()
        estimates.append(est)

    mean_est = sum(estimates) / N_TRIALS
    # Allow 1% absolute error on the mean.
    assert abs(mean_est - true_ip) < 0.01 * max(1.0, abs(true_ip)), (
        f"E[ProdQJL]={mean_est:.4f} ≠ ⟨q,k⟩={true_ip:.4f}"
    )


def test_estimator_batched_keys():
    S = make_projection(D, M, seed=99)
    q = torch.randn(D)
    keys = torch.randn(8, D)
    k_tilde, nu = qjl_encode(S, keys)
    estimates = prod_qjl(S, q, k_tilde, nu)
    assert estimates.shape == (8,)


def test_projection_reproducible():
    S1 = make_projection(D, M, seed=7)
    S2 = make_projection(D, M, seed=7)
    assert torch.equal(S1, S2)
