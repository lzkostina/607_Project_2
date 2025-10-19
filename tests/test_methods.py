import numpy as np
import pytest

from src.dgps import generate_design
from src.methods import knockoffs_equicorr

def test_knockoffs_shapes_and_gram_identities():
    # Use n >= 2p
    n, p = 240, 100
    X = generate_design(n=n, p=p, mode="iid", seed=123, normalize=True, norm_target="sqrt_n")

    X_knock, meta = knockoffs_equicorr(X, seed=7)

    # Shapes
    assert X_knock.shape == (n, p)

    # Gram matrices
    XTX = X.T @ X
    XTXk = X.T @ X_knock
    XkTXk = X_knock.T @ X_knock

    # Sigma in construction is (X^T X) / n; identities become:
    # X^T Xk ≈ X^T X - n S  and  Xk^T Xk ≈ X^T X
    S = meta["S"]  # s * I
    assert S.shape == (p, p)

    # Tolerances: allow small numerical error
    atol = 1e-6 * n  # scales mildly with n
    rtol = 1e-6

    # Check X^T Xk + n S ≈ X^T X
    assert np.allclose(XTXk + n * S, XTX, rtol=rtol, atol=atol)

    # Check Xk^T Xk ≈ X^T X
    assert np.allclose(XkTXk, XTX, rtol=rtol, atol=atol)


def test_knockoffs_determinism_and_variation_with_seed():
    n, p = 220, 110  # still n >= 2p
    X = generate_design(n=n, p=p, mode="iid", seed=1, normalize=True, norm_target="sqrt_n")

    X_knock_1, meta_1 = knockoffs_equicorr(X, seed=2024)
    X_knock_2, meta_2 = knockoffs_equicorr(X, seed=2024)
    X_knock_3, meta_3 = knockoffs_equicorr(X, seed=2025)

    # Same seed -> identical knockoffs and identical S/A/M
    assert np.array_equal(X_knock_1, X_knock_2)
    assert np.allclose(meta_1["S"], meta_2["S"])
    assert np.allclose(meta_1["A"], meta_2["A"])
    assert np.allclose(meta_1["M"], meta_2["M"])

    # Different seed -> typically different Xk (orthogonal complement randomness)
    assert not np.array_equal(X_knock_1, X_knock_3)


def test_knockoffs_invalid_inputs():
    # Here n < 2p should raise
    n, p = 150, 100  # n - p = 50 < p
    X = generate_design(n=n, p=p, mode="iid", seed=3, normalize=True, norm_target="sqrt_n")

    with pytest.raises(ValueError):
        _ = knockoffs_equicorr(X, seed=0)


def test_knockoffs_s_and_M_properties():
    n, p = 260, 120
    X = generate_design(n=n, p=p, mode="iid", seed=7, normalize=True, norm_target="sqrt_n")

    X_knock, meta = knockoffs_equicorr(X, seed=9)

    s = meta["s"]
    M = meta["M"]

    # s should be in [0, 1] (up to tiny numerical tolerance)
    assert s >= -1e-10 and s <= 1.0 + 1e-10

    # M should be PSD (all eigenvalues >= -tiny_tol)
    evals = np.linalg.eigvalsh(0.5 * (M + M.T))
    assert evals.min() >= -1e-9


