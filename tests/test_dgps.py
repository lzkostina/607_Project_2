import numpy as np
import pytest
import math

from src.dgps import auto_regressive_cov, generate_design, generate_errors

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


def test_generate_errors_student_t_rescaled_and_sigma2_validation():
    n = 3000
    df = 5.0
    z = generate_errors(n=n, df=df, sigma2=1.0, seed=999)

    # approximate zero mean and unit variance after rescaling
    assert abs(z.mean()) < 0.05
    assert abs(z.var(ddof=1) - 1.0) < 0.05

    # sigma2 must be exactly 1.0 (within tiny tolerance) -> any other value should raise
    with pytest.raises(ValueError):
        _ = generate_errors(n=n, df=df, sigma2=0.9, seed=1)

