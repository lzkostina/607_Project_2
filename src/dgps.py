import numpy as np
import warnings
import math

EPS = 1e-12


def auto_regressive_cov(p: int, rho: float) -> np.ndarray:
    """
    Construct a covariance matrix Sigma with entries:
        Sigma[j, k] = rho ** ||j - k||,   -1 < rho < 1

    Parameters:
    ----------
    p : int
        Dimensionality (number of predictors)
    rho : float
        Autocorrelation parameter in [0, 1)

    Returns:
    --------
    Sigma : (p, p) ndarray (float64)
    """

    if not isinstance(p, int) or p < 1:
        raise ValueError("p must be a positive integer")
    if not (-1 < rho < 1):
        raise ValueError("rho must be in (-1, 1)")
    idx = np.arange(p)

    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_design(
    n: int,
    p: int,
    mode: str = "iid",              # {"iid", "ar1"}
    rho: float | None = None,       # required iff mode == "ar1"
    seed: int | None = None,
    normalize: bool = True,         # column-wise normalization
    norm_target: str = "sqrt_n"     # {"sqrt_n", "unit_var"}
) -> np.ndarray:
    """
        Generate a design matrix X of shape (n, p).

        Modes
        -----
        - mode="iid": rows iid ~ N(0, I_p)   (rho is ignored)
        - mode="ar1": rows iid ~ N(0, Σ_rho) with (Σ_rho)_{jk} = rho^{|j-k|}, -1 < rho < 1

        Normalization (applied column-wise if normalize=True)
        -----------------------------------------------------
        - norm_target="sqrt_n": scale each column so ||X[:, j]||_2 = sqrt(n)
        - norm_target="unit_var": center and scale each column to sample variance ~ 1 (ddof=1)

        Determinism
        -----------
        Using the same (n, p, mode, rho, seed, normalize, norm_target) yields identical X.

        Parameters
        ----------
        n : int
            Number of observations (n > 0).
        p : int
            Number of predictors (p > 0).
        mode : {"iid", "ar1"}
            Design type.
        rho : float | None
            Autocorrelation parameter for AR(1). Must satisfy -1 < rho < 1 when mode="ar1".
            Ignored when mode="iid".
        seed : int | None
            Seed for the RNG.
        normalize : bool
            Whether to normalize columns (recommended True for comparability across trials).
        norm_target : {"sqrt_n", "unit_var"}
            Target normalization convention.

        Returns
        -------
        X : np.ndarray, shape (n, p), dtype float64
    """
    # inputs check
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(p, int) or p < 1:
        raise ValueError("p must be a positive integer")

    if mode not in {"iid", "ar1"}:
        raise ValueError("mode must be one of {'iid', 'ar1'}.")
    if norm_target not in {"sqrt_n", "unit_var"}:
        raise ValueError("norm_target must be one of {'sqrt_n', 'unit_var'}.")
    if mode == "ar1":
        if rho is None:
            raise ValueError("rho must be provided when mode='ar1'.")
        if not (-1 < rho < 1):
            raise ValueError("For AR(1), rho must be in (-1, 1).")
    else:
        if rho is not None:
            warnings.warn("rho provided but will be ignored for mode='iid'.", RuntimeWarning)

    rng = np.random.default_rng(seed)

    if mode == "iid":
        X = rng.standard_normal(size=(n, p), dtype=np.float64)
    else:
        # Build AR(1) covariance and its (stabilized) Cholesky factor
        Sigma = auto_regressive_cov(p, rho)
        L = np.linalg.cholesky(Sigma + EPS * np.eye(p, dtype=np.float64))
        Z = rng.standard_normal(size=(n, p), dtype=np.float64)
        X = Z @ L.T

    X = np.asarray(X, dtype=np.float64, order="C")

    if normalize:
        if norm_target == "sqrt_n":
            # scale columns so ||X[:, j]||_2 = sqrt(n)
            col_norms = np.linalg.norm(X, axis=0)
            # avoid division by zero for any pathological zero-norm column
            safe_norms = np.where(col_norms <= EPS, 1.0, col_norms)
            X = X * (np.sqrt(n) / safe_norms)
        else:
            # center then scale to sample variance ~ 1
            X = X - X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, ddof=1, keepdims=True)
            sd = np.where(sd <= EPS, 1.0, sd)
            X = X / sd

    return X
