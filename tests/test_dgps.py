import numpy as np
import pytest
import math

from src.dgps import auto_regressive_cov, generate_design

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
