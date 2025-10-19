import numpy as np
import pytest

from src.dgps import generate_design, generate_full
from src.methods import knockoffs_equicorr, lasso_path_stats, knockoff_threshold, knockoff_select

##################################  knockoffs_equicorr tests  #######################

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


##################################  lasso_path_stats tests  #######################

def _make_data(n=300, p=120, k=10, A=6.0, seed=123, mode="iid"):
    # n >= 2p needed by build_knockoffs
    assert n >= 2 * p, "Test helper requires n >= 2p."
    y, X, beta, meta = generate_full(
        n, p,
        mode=mode,
        k=k, A=A,
        seed=seed
    )
    Xk, _ = knockoffs_equicorr(X, seed=seed + 1)
    return X, y, Xk, beta


def test_lasso_path_stats_shapes_and_alphas():
    n, p = 260, 120  # n >= 2p
    X, y, X_knock, _ = _make_data(n=n, p=p, k=8, A=5.0, seed=7)

    out = lasso_path_stats(X, y, X_knock, n_alphas=120, eps=1e-3, coef_tol=1e-9)

    # shapes
    assert out["W"].shape == (p,)
    assert out["Z_orig"].shape == (p,)
    assert out["Z_knock"].shape == (p,)
    assert out["coefs"].shape[0] == 2 * p
    assert out["coefs"].shape[1] == out["alphas"].shape[0]

    # alphas are strictly decreasing
    alphas = out["alphas"]
    assert np.all(np.diff(alphas) < 0)


def test_lasso_path_stats_determinism():
    n, p = 240, 110
    X, y, Xk, _ = _make_data(n=n, p=p, k=6, A=5.0, seed=11)

    meta_1 = lasso_path_stats(X, y, Xk, n_alphas=100, eps=1e-3, coef_tol=1e-9)
    meta_2 = lasso_path_stats(X, y, Xk, n_alphas=100, eps=1e-3, coef_tol=1e-9)

    # identical outputs for identical inputs
    assert np.array_equal(meta_1["W"], meta_2["W"])
    assert np.array_equal(meta_1["Z_orig"], meta_2["Z_orig"])
    assert np.array_equal(meta_1["Z_knock"], meta_2["Z_knock"])
    assert np.array_equal(meta_1["alphas"], meta_2["alphas"])
    assert np.array_equal(meta_1["coefs"], meta_2["coefs"])


def test_lasso_path_stats_zero_response_all_zero():
    n, p = 220, 100
    # Build X, Xk; then set y to zeros to force all coefs to zero along path
    _, X, _, _ = generate_full(n, p, mode="iid", k=0, A=0.0, seed=5)  # use helper to get normalized X
    Xk, _ = knockoffs_equicorr(X, seed=13)
    y = np.zeros(n, dtype=float)

    out = lasso_path_stats(X, y, Xk, n_alphas=80, eps=1e-3, coef_tol=1e-9)

    assert np.allclose(out["Z_orig"], 0.0)
    assert np.allclose(out["Z_knock"], 0.0)
    assert np.allclose(out["W"], 0.0)


def test_lasso_path_stats_signal_sanity_more_positive_W():
    n, p = 280, 120
    # strong sparse signal -> originals should tend to win (W > 0)
    X, y, Xk, beta = _make_data(n=n, p=p, k=12, A=7.0, seed=33)

    out = lasso_path_stats(X, y, Xk, n_alphas=120, eps=1e-3, coef_tol=1e-9)

    n_pos = int((out["W"] > 0).sum())
    n_neg = int((out["W"] < 0).sum())

    # Heuristic check: there should be more positives than negatives
    assert n_pos > n_neg

############################# knockoff_threshold tests #######################

def test_knockoff_threshold_basic():
    # Construct W with many positive, few small negatives.
    # With q=0.2 and offset=1 (Knockoff+), the smallest feasible t is 0.8:
    #   candidates t: 0.1, 0.2, 0.8, 1.0, 1.5, 2.0, 3.0
    #   at t=0.8: num = 1 + #{W <= -0.8} = 1 + 0 = 1
    #             den = max(1, #{W >= 0.8}) = 5
    #             fdp_hat = 1/5 = 0.2 <= q
    W = np.array([3.0, 2.0, 1.5, 1.0, 0.8, -0.2, -0.1, 0.0], dtype=float)
    q = 0.2
    T = knockoff_threshold(W, q=q, offset=1)
    assert np.isfinite(T)
    assert abs(T - 0.8) < 1e-12  # exact in this synthetic setup


############################# knockoff_select tests #######################
def test_knockoff_select_basic():
    # Same W as above: expect selection of indices with W >= 0.8
    W = np.array([3.0, 2.0, 1.5, 1.0, 0.8, -0.2, -0.1, 0.0], dtype=float)
    q = 0.2
    selected, info = knockoff_select(W, q=q, offset=1)

    # Expected indices: 0..4
    assert np.array_equal(selected, np.array([0, 1, 2, 3, 4], dtype=int))

    # Threshold and FDP estimate match the hand calculation
    assert np.isfinite(info["T"]) and abs(info["T"] - 0.8) < 1e-12
    assert abs(info["fdp_hat"] - 0.2) < 1e-12

    # Sanity on counts used in FDP calculation
    assert info["num_neg"] == 0   # #{W <= -T} with T=0.8
    assert info["num_pos"] == 5   # #{W >= T} with T=0.8
