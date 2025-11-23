import numpy as np
import warnings
import math
from typing import Optional, Sequence


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
    #OLD VERSION
    # if not math.isfinite(df) and df is not math.inf:
    #     raise ValueError("df must be positive or math.inf.")
    # if math.isfinite(df) and df <= 0:
    #     raise ValueError("df must be positive when finite.")
    # NEW VERSION
    # we need it for joblib
    df = float(df)  # ensure scalar

    # Accept df = math.inf or df = np.inf or df > 0
    if not (df > 0.0 or math.isinf(df)):
        raise ValueError("df must be positive or math.inf.")

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

def scale_cols_unit_l2(X: np.ndarray) -> np.ndarray:
    """
    Center columns, then scale so each column has ||X_j||_2 = 1 (not sqrt(n)).
    This makes Var(X_j) ≈ 1/n and keeps SNR moderate with A fixed.
    """
    X = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=0, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    X = X / norms
    return X

def generate_full(
    n: int,
    p: int,
    *,
    mode: str = "iid",                 # {"iid", "ar1"}
    rho: float | None = None,          # required iff mode == "ar1"
    df: float = math.inf,              # default Gaussian noise
    # --- signal (paper-faithful defaults) ---
    k: int = 30,                       # number of nonzeros in beta
    A: float = 3.5,                    # amplitude of each nonzero
    sign_mode: str = "random",         # {"random", "positive_only"}
    support_indices: Optional[Sequence[int]] = None,  # optional fixed support
    # --- design normalization ---
    normalize: bool = True,
    norm_target: str = "sqrt_n",       # {"sqrt_n", "unit_var"}
    # --- RNG ---
    seed: int | None = None,
    # --- disallowed for this project (kept for explicit error) ---
    snr: float | None = None,          # not supported with sigma2=1.0 enforced
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate (y, X, beta) for the linear model y = X beta + z with unit-variance noise.

    This implementation is tailored to the knockoffs reproduction:
      - Design X: iid or AR(1) with column-wise normalization (default on)
      - Beta: exactly k nonzeros with fixed amplitude ±A (random signs by default)
      - Noise: Gaussian (df=inf) with variance 1.0 (enforced by generate_errors)

    Parameters
    ----------
    n, p : int
        Sample size and number of predictors (must be positive).
    mode : {"iid", "ar1"}
        Design type.
    rho : float | None
        AR(1) parameter (-1 < rho < 1). Required iff mode == "ar1".
    df : float
        Degrees of freedom for noise; math.inf => Gaussian N(0,1).
    k : int
        Number of nonzero coefficients in beta (1 <= k <= p).
    A : float
        Amplitude for each nonzero coefficient (A >= 0).
    sign_mode : {"random", "positive_only"}
        If "random", each nonzero is +A or -A with prob 1/2.
        If "positive_only", all nonzeros are +A.
    support_indices : sequence[int] | None
        If provided, must contain exactly k distinct indices in [0, p-1].
        Otherwise a random support of size k is drawn.
    normalize : bool
        If True, normalize columns of X (recommended).
    norm_target : {"sqrt_n", "unit_var"}
        Column-wise normalization target.
    seed : int | None
        Master seed. Internally split to keep design/beta/noise independent.
    snr : float | None
        Not supported here (noise variance fixed at 1.0). If not None, raises.

    Returns
    -------
    y : (n,) ndarray
    X : (n, p) ndarray
    beta : (p,) ndarray
    meta : dict with keys:
        - mode, rho, df
        - k, A, sign_mode
        - support_indices (list[int])
        - n_pos, n_neg
        - normalize, norm_target
        - seed, seed_design, seed_beta, seed_noise
        - empirical_snr = ||X beta||^2 / ||z||^2
        - col_norm_min, col_norm_max  (after normalization)
    """
    # ---- validation ----
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer.")
    if not isinstance(p, int) or p < 1:
        raise ValueError("p must be a positive integer.")
    if not isinstance(k, int) or not (0 <= k <= p):
        raise ValueError("k must be an integer in [0, p].")
    if A < 0:
        raise ValueError("A must be nonnegative.")
    if sign_mode not in {"random", "positive_only"}:
        raise ValueError("sign_mode must be one of {'random', 'positive_only'}.")
    if snr is not None:
        raise ValueError("snr-targeting not supported (noise variance fixed at 1.0).")
    if mode not in {"iid", "ar1"}:
        raise ValueError("mode must be one of {'iid', 'ar1'}.")

    # ---- master RNG and sub-seeds (independence across components) ----
    rng_master = np.random.default_rng(seed)
    seed_design = int(rng_master.integers(2**31 - 1))
    seed_beta   = int(rng_master.integers(2**31 - 1))
    seed_noise  = int(rng_master.integers(2**31 - 1))

    # ---- 1) Design ----
    X = generate_design(
        n=n, p=p, mode=mode, rho=rho, seed=seed_design,
        normalize=normalize, norm_target=norm_target
    )
    X = scale_cols_unit_l2(X)

    # diagnostics for meta
    col_norms = np.linalg.norm(X, axis=0)
    col_norm_min = float(np.min(col_norms))
    col_norm_max = float(np.max(col_norms))

    # ---- 2) Beta (exactly k nonzeros with fixed amplitude) ----
    beta = np.zeros(p, dtype=np.float64)
    rng_beta = np.random.default_rng(seed_beta)

    if k > 0:
        if support_indices is not None:
            supp = np.array(sorted(set(support_indices)), dtype=int)
            if supp.size != k:
                raise ValueError("support_indices must contain exactly k distinct indices.")
            if supp.min() < 0 or supp.max() >= p:
                raise ValueError("support_indices out of bounds.")
        else:
            supp = rng_beta.choice(p, size=k, replace=False)

        if sign_mode == "random":
            signs = rng_beta.choice(np.array([-1.0, 1.0]), size=k)
        else:  # "positive_only"
            signs = np.ones(k, dtype=np.float64)

        beta[supp] = signs * float(A)
        n_pos = int(np.sum(beta[supp] > 0))
        n_neg = int(np.sum(beta[supp] < 0))
    else:
        supp = np.array([], dtype=int)
        n_pos = 0
        n_neg = 0

    # ---- 3) Noise (variance fixed at 1.0 via generate_errors) ----
    z = generate_errors(n=n, df=df, sigma2=1.0, seed=seed_noise)

    # ---- 4) Response ----
    y = X @ beta + z

    # ---- 5) Meta ----
    signal_energy = float(beta @ (X.T @ (X @ beta)))
    noise_energy  = float(z @ z)
    empirical_snr = math.inf if noise_energy <= EPS else signal_energy / noise_energy

    meta = dict(
        mode=mode,
        rho=None if mode == "iid" else float(rho),
        df=float(df),
        k=int(k),
        A=float(A),
        sign_mode=sign_mode,
        support_indices=supp.tolist(),
        n_pos=n_pos,
        n_neg=n_neg,
        normalize=bool(normalize),
        norm_target=norm_target,
        seed=None if seed is None else int(seed),
        seed_design=seed_design,
        seed_beta=seed_beta,
        seed_noise=seed_noise,
        empirical_snr=float(empirical_snr),
        col_norm_min=col_norm_min,
        col_norm_max=col_norm_max,
    )

    return y, X, beta, meta








