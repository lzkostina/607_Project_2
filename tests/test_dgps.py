import numpy as np
import pytest
import math

from src.dgps import auto_regressive_cov, generate_design, generate_errors, generate_full

####################### auto_regressive_cov tests #######################

def test_ar1_cov_shape_and_diagonal():
    S = auto_regressive_cov(5, 0.5)
    assert S.shape == (5, 5)
    assert np.allclose(np.diag(S), 1.0)


def test_ar1_cov_values_positive_rho():
    rho = 0.5
    S = auto_regressive_cov(4, rho)
    expected = np.array([
        [1.0, rho, rho**2, rho**3],
        [rho, 1.0, rho, rho**2],
        [rho**2, rho, 1.0, rho],
        [rho**3, rho**2, rho, 1.0],
    ])
    assert np.allclose(S, expected)


def test_ar1_cov_values_negative_rho():
    rho = -0.7
    S = auto_regressive_cov(4, rho)
    expected = np.array([
        [1.0, rho, rho**2, rho**3],
        [rho, 1.0, rho, rho**2],
        [rho**2, rho, 1.0, rho],
        [rho**3, rho**2, rho, 1.0],
    ])
    assert np.allclose(S, expected)


def test_ar1_cov_rho_bounds():
    # Now allow -1 < rho < 1
    with pytest.raises(ValueError):
        auto_regressive_cov(3, -1.0)
    with pytest.raises(ValueError):
        auto_regressive_cov(3, 1.0)
    with pytest.raises(ValueError):
        auto_regressive_cov(0, 0.5)  # p must be positive int


def test_ar1_cov_positive_definite_small_p():
    # For -1 < rho < 1, AR(1) covariance is PD; Cholesky should succeed (with tiny jitter)
    for rho in (0.0, 0.2, 0.8, 0.999, -0.2, -0.8, -0.999):
        S = auto_regressive_cov(8, rho)
        L = np.linalg.cholesky(S + 1e-12 * np.eye(S.shape[0]))
        recon = L @ L.T
        assert np.allclose(recon, S + 1e-12 * np.eye(S.shape[0]), rtol=1e-6, atol=1e-8)

########################## generate_design tests ###########################

def test_generate_design_shapes_and_dtype_iid():
    X = generate_design(n=50, p=20, mode="iid", rho=None, seed=123)
    assert X.shape == (50, 20)
    assert X.dtype == np.float64


def test_generate_design_shapes_and_dtype_ar1():
    X = generate_design(n=60, p=15, mode="ar1", rho=0.6, seed=42)
    assert X.shape == (60, 15)
    assert X.dtype == np.float64


def test_generate_design_normalization_sqrt_n():
    n, p = 80, 17
    X = generate_design(n=n, p=p, mode="iid", rho=None, seed=7,
                        normalize=True, norm_target="sqrt_n")
    norms = np.linalg.norm(X, axis=0)
    assert np.allclose(norms, np.sqrt(n), rtol=1e-10, atol=1e-10)


def test_generate_design_normalization_unit_var():
    n, p = 120, 10
    X = generate_design(n=n, p=p, mode="iid", rho=None, seed=9,
                        normalize=True, norm_target="unit_var")
    # centered (approximately) and unit sample variance
    col_means = X.mean(axis=0)
    col_vars = X.var(axis=0, ddof=1)
    assert np.allclose(col_means, 0.0, atol=5e-2)   # loose tolerance due to randomness
    assert np.allclose(col_vars, 1.0, atol=5e-2)


def test_generate_design_determinism():
    X1 = generate_design(n=40, p=12, mode="iid", rho=None, seed=2024)
    X2 = generate_design(n=40, p=12, mode="iid", rho=None, seed=2024)
    assert np.array_equal(X1, X2)


def test_generate_design_different_seeds_change_output():
    X1 = generate_design(n=40, p=12, mode="iid", rho=None, seed=1)
    X2 = generate_design(n=40, p=12, mode="iid", rho=None, seed=2)
    # With high probability, they differ; if not, check a statistic
    assert not np.array_equal(X1, X2)


def test_generate_design_ar1_empirical_lag1_corr_matches_rho_sign():
    n, p, rho = 400, 50, 0.7
    X = generate_design(n=n, p=p, mode="ar1", rho=rho, seed=11)
    # empirical column correlation matrix
    C = np.corrcoef(X, rowvar=False)
    # average lag-1 correlation across adjacent columns
    lag1 = np.mean([C[j, j+1] for j in range(p-1)])
    assert lag1 > 0.4  # coarse check: should be noticeably positive

    # negative rho
    rho = -0.7
    Xn = generate_design(n=n, p=p, mode="ar1", rho=rho, seed=11)
    Cn = np.corrcoef(Xn, rowvar=False)
    lag1n = np.mean([Cn[j, j+1] for j in range(p-1)])
    assert lag1n < -0.4  # should be noticeably negative


def test_generate_design_invalid_mode_and_rho_requirements():
    with pytest.raises(ValueError):
        generate_design(n=10, p=5, mode="weird", rho=None)
    # ar1 requires rho
    with pytest.raises(ValueError):
        generate_design(n=10, p=5, mode="ar1", rho=None)
    # ar1 rho bounds
    with pytest.raises(ValueError):
        generate_design(n=10, p=5, mode="ar1", rho=1.0)
    with pytest.raises(ValueError):
        generate_design(n=10, p=5, mode="ar1", rho=-1.0)


def test_generate_design_iid_ignores_rho():
    # Ensure iid runs even if rho is set (you may emit a warning internally)
    X = generate_design(n=30, p=8, mode="iid", rho=0.9, seed=5)
    assert X.shape == (30, 8)


############################# generate_errors tests ###########################

def test_generate_errors_gaussian_determinism_and_unit_var():
    n = 2000
    z1 = generate_errors(n=n, df=math.inf, sigma2=1.0, seed=123)
    z2 = generate_errors(n=n, df=math.inf, sigma2=1.0, seed=123)
    z3 = generate_errors(n=n, df=math.inf, sigma2=1.0, seed=456)

    # shape and determinism
    assert z1.shape == (n,)
    assert np.array_equal(z1, z2)
    assert not np.array_equal(z1, z3)

    # approximate zero mean and unit variance
    assert abs(z1.mean()) < 0.05          # CLT tolerance for n=2000
    assert abs(z1.var(ddof=1) - 1.0) < 0.05

####################### generate_full tests ###########################

def test_generate_full_shapes_and_determinism_iid():
    n, p = 60, 25
    y1, X1, b1, m1 = generate_full(n, p, mode="iid", k=5, A=3.5, seed=123)
    y2, X2, b2, m2 = generate_full(n, p, mode="iid", k=5, A=3.5, seed=123)
    y3, X3, b3, m3 = generate_full(n, p, mode="iid", k=5, A=3.5, seed=124)

    assert X1.shape == (n, p)
    assert y1.shape == (n,)
    assert b1.shape == (p,)

    # determinism
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(b1, b2)
    # likely different with different seed
    assert not np.array_equal(X1, X3) or not np.array_equal(y1, y3)


def test_generate_full_beta_sparsity_and_amplitudes_random_signs():
    n, p, k, A = 80, 40, 7, 2.25
    y, X, beta, meta = generate_full(n, p, mode="iid", k=k, A=A, seed=7)

    nnz_idx = np.flatnonzero(beta)
    assert nnz_idx.size == k
    # all nonzeros are exactly ±A
    assert np.all(np.isin(np.unique(np.abs(beta[nnz_idx])), [A]))
    # sign counts in meta
    assert meta["n_pos"] + meta["n_neg"] == k
    # support indices in meta match beta's support (order-insensitive)
    assert set(meta["support_indices"]) == set(nnz_idx.tolist())


def test_generate_full_fixed_support_and_positive_only():
    n, p, k, A = 50, 30, 5, 3.5
    support = [0, 3, 5, 9, 12]
    y, X, beta, meta = generate_full(
        n, p, mode="iid", k=k, A=A,
        sign_mode="positive_only",
        support_indices=support,
        seed=42
    )
    nnz_idx = np.flatnonzero(beta)
    assert set(nnz_idx.tolist()) == set(support)
    assert np.all(beta[nnz_idx] == A)  # all positive
    assert meta["n_neg"] == 0 and meta["n_pos"] == k


def test_generate_full_ar1_empirical_lag1_corr_sign():
    n, p, rho = 300, 35, 0.8
    _, X_pos, _, _ = generate_full(n, p, mode="ar1", rho=rho, k=0, A=0.0, seed=11)
    C_pos = np.corrcoef(X_pos, rowvar=False)
    lag1_pos = np.mean([C_pos[j, j+1] for j in range(p-1)])
    assert lag1_pos > 0.4  # noticeably positive

    rho = -0.8
    _, X_neg, _, _ = generate_full(n, p, mode="ar1", rho=rho, k=0, A=0.0, seed=11)
    C_neg = np.corrcoef(X_neg, rowvar=False)
    lag1_neg = np.mean([C_neg[j, j+1] for j in range(p-1)])
    assert lag1_neg < -0.4  # noticeably negative


def test_generate_full_snr_not_supported_raises():
    with pytest.raises(ValueError):
        _ = generate_full(40, 15, mode="iid", k=3, A=1.0, snr=5.0, seed=1)


def test_generate_full_k_zero_zero_beta_and_unit_noise_variance():
    n, p = 500, 25
    y, X, beta, meta = generate_full(n, p, mode="iid", k=0, A=0.0, seed=99)
    # beta is all zeros
    assert np.all(beta == 0.0)
    # y should essentially be pure noise with unit variance
    # (mean ~ 0, sample variance ~ 1 with tolerance)
    assert abs(y.mean()) < 0.1
    assert abs(y.var(ddof=1) - 1.0) < 0.1
