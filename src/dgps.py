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


def generate_errors(
    n: int,
    df: float = math.inf,
    sigma2: float = 1.0,
    seed: int | None = None
) -> np.ndarray:
    """
    Generate an error vector z of length n with unit variance.

    Default is Gaussian N(0, 1). If df is finite, draws from Student-t(df),
    recenters, and rescales to have sample variance ≈ 1.

    Parameters
    ----------
    n : int
        Number of observations (must be > 0).
    df : float, default: math.inf
        Degrees of freedom. Use math.inf for Gaussian; else df > 0 for t.
    sigma2 : float, default: 1.0
        Target variance. For this project we only support sigma2 == 1.0.
    seed : int | None
        Random seed for determinism.

    Returns
    -------
    eps : np.ndarray, shape (n,), dtype float64
        Zero-mean (approximately), unit-variance noise.
    """
    # ---- validation ----
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not math.isfinite(df) and df is not math.inf:
        raise ValueError("df must be positive or math.inf.")
    if math.isfinite(df) and df <= 0:
        raise ValueError("df must be positive when finite.")
    # enforce unit variance for this reproduction (keep it strict but tolerant)
    if abs(float(sigma2) - 1.0) > 1e-12:
        raise ValueError("sigma2 must be 1.0 for this project.")

    rng = np.random.default_rng(seed)

    # ---- Gaussian branch ----
    if df is math.inf:
        return rng.standard_normal(size=n).astype(np.float64)

    # ---- Student-t branch (rescale to unit variance) ----
    t = rng.standard_t(df, size=n).astype(np.float64)

    # center
    t -= t.mean()

    # scale to sample variance ≈ 1 (guard against pathological zero variance)
    cur_var = t.var(ddof=1)
    if cur_var <= EPS:
        # extremely unlikely; fall back to Gaussian(0,1)
        return rng.standard_normal(size=n).astype(np.float64)

    scale = 1.0 / math.sqrt(cur_var)
    eps = t * scale
    return eps


def generate_full(n: int, p: int, rho: float, df: float,
    sigma2: float | None = None,
    snr: float | None = None,      # target SNR = (beta^T X^T X beta) / sigma2
    beta_sparsity: int | None = None,  # None = dense; else exactly k nonzeros
    beta_scale: float = 1.0,       # std of nonzero beta entries
    center_X: bool = False,
    standardize_X: bool = False,
    seed: int | None = None,
    ):
    """
    Generate (y, X, beta) for y = X beta + eps.

    Models:
    -----
    - Fixed noise: pass sigma2 (snr can be None).
    - SNR targeting: pass snr (sigma2 can be None), we set
        sigma2 = (beta^T X^T X beta) / snr
      using the realized X and beta.

    Returns:
    -------
    y : (n, ) ndarray
    X : (n, p) ndarray
    beta : (p, ) ndarray
    meta : dict  (sigma2, empirical_snr, df, rho, snr_target, seed, etc.)

    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")

    if not isinstance(p, int) or p <= 0:
        raise ValueError("p must be a positive integer.")

    if sigma2 is None and snr is None:
        raise ValueError("Provide either sigma2 or snr.")

    if sigma2 is not None and sigma2 < 0:
        raise ValueError("sigma2 must be nonnegative.")

    if snr is not None and snr <= 0:
        raise ValueError("snr must be positive.")

    rng = np.random.default_rng(seed)
    # Split the master RNG stream so design and errors aren’t coupled
    seed_X  = int(rng.integers(2**31 - 1))
    seed_eps = int(rng.integers(2**31 - 1))

    # 1) Design
    X = generate_data(n=n, p=p, rho=rho, seed=seed_X)

    if center_X or standardize_X:
        X = X.astype(float, copy=True)
        if center_X:
            X -= X.mean(axis=0, keepdims=True)
        if standardize_X:
            sd = X.std(axis=0, ddof=1, keepdims=True)
            sd = np.where(sd <= EPS, 1.0, sd)
            X /= sd

    # 2) Beta (dense or k-sparse)
    if beta_sparsity is None:
        beta = rng.normal(loc=0.0, scale=beta_scale, size=p)
    else:
        if not (1 <= beta_sparsity <= p):
            raise ValueError("beta_sparsity must be in [1, p].")
        beta = np.zeros(p)
        idx = rng.choice(p, size=beta_sparsity, replace=False)
        beta[idx] = rng.normal(loc=0.0, scale=beta_scale, size=beta_sparsity)

    # 3) Choose sigma2 (fixed or via SNR)
    if sigma2 is None:
        # SNR targeting: sigma2 = (beta^T X^T X beta) / snr
        signal_energy = float(beta @ (X.T @ (X @ beta)))
        if signal_energy <= EPS:
            sigma2 = 1.0  # degenerate case: give reasonable noise
        else:
            sigma2 = signal_energy / float(snr)

    # 4) Errors
    eps = generate_errors(n=n, df=df, sigma2=float(sigma2), seed=seed_eps)

    # 5) Response
    y = X @ beta + eps

    # 6) Meta info
    noise_var = float(eps.var(ddof=1))
    signal_energy = float(beta @ (X.T @ (X @ beta)))
    empirical_snr = math.inf if noise_var <= EPS else signal_energy / noise_var

    meta = dict(
        sigma2=float(sigma2),
        empirical_snr=float(empirical_snr),
        df=float(df),
        rho=float(rho),
        snr_target=(None if snr is None else float(snr)),
        seed=int(seed) if seed is not None else None,
        center_X=bool(center_X),
        standardize_X=bool(standardize_X),
        beta_sparsity=(None if beta_sparsity is None else int(beta_sparsity)),
    )
    return y, X, beta, meta







