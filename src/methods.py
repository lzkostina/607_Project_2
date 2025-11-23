import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.linear_model import lars_path
from scipy import stats
from scipy.stats import norm
from typing import Dict
import warnings

def knockoffs_equicorr(X: np.ndarray, *, use_true_Sigma: np.ndarray | None = None,seed: int | None = None
        )->tuple[np.ndarray, dict]:
    """
        Construct equi-correlated knockoffs X_knock for a fixed design matrix X.

    Parameters:
    ----------
    X : ndarray of shape (n, p)
        Design matrix.
    use_true_Sigma : ndarray of shape (p, p) or None
        If provided, uses this Sigma instead of (X^T X) / n.
    seed : int or None
        RNG seed (used for the orthogonal complement basis and its column orthonormalization).

    Returns:
    -------
    X_knock : ndarray of shape (n, p)
        Knockoff matrix.
    meta : dict
        Meta information.
    """

    X = np.asarray(X, dtype=np.float64, order="C")
    n, p = X.shape

    # input checks
    if n <= 0 or p <= 0:
        raise ValueError("X must have positive shape (n, p).")
    if n - p < p:
        raise ValueError(
            f"Construction requires n - p >= p (i.e., n >= 2p). Got n={n}, p={p}."
        )
    if use_true_Sigma is None:
        # Use sample correlation-like Gram (since columns are normalized, diag ~ 1)
        Sigma = (X.T @ X) / float(n)
    else:
        Sigma = np.array(use_true_Sigma, dtype=float, copy=True)
        if Sigma.shape != (p, p):
            raise ValueError("use_true_Sigma must have shape (p, p).")

    # Symmetrize
    Sigma += Sigma.T
    Sigma *= 0.5

    # numerical stability
    evals, evecs = np.linalg.eigh(Sigma)
    lambda_min = float(np.min(evals))

    if lambda_min < -1e-8:
        raise ValueError(f"Sigma appears indefinite: min eigenvalue = {lambda_min:.3e}")

    evals_clamped = np.maximum(evals, 0.0)
    Sigma_psd = (evecs * evals_clamped) @ evecs.T  # rebuild PSD Sigma

    lambda_min_clamped = float(np.min(evals_clamped))
    s = min(1.0, 2.0 * lambda_min_clamped)
    s = max(0.0, float(s) - 1e-12)

    S = s * np.eye(p, dtype=np.float64) # S = s I, with s <= min(1, 2*lambda_min)

    # safe inverse
    ridge = 0.0
    if lambda_min_clamped < 1e-10:
        ridge = 1e-8
    Sigma_inv = np.linalg.solve(Sigma_psd + ridge * np.eye(p), np.eye(p))

    A = np.eye(p, dtype=np.float64) - (Sigma_inv @ S)  # A = I - Sigma^{-1} S  (p x p)

    # M = n * (2S - S Sigma^{-1} S)  (p x p), PSD by construction
    M = n * (2.0 * S - (S @ Sigma_inv @ S))

    # Symmetrize
    M = 0.5 * (M + M.T)

    # --- PSD square root of M (p x p) via eigen-decomposition ---
    m_evals, m_evecs = np.linalg.eigh(M)
    m_evals_clamped = np.maximum(m_evals, 0.0)

    # Numerical rank (for info)
    rank_M = int(np.sum(m_evals_clamped > 1e-12))
    sqrtM = (m_evecs * np.sqrt(m_evals_clamped)) @ m_evecs.T  # (p x p)

    # --- Build orthonormal basis for the orthogonal complement of col(X) ---
    # Qx: n x p with orthonormal columns spanning col(X)
    Qx, _ = np.linalg.qr(X, mode="reduced")  # economy QR
    # Random matrix G in R^{n x (n-p)}; project out components in col(X)
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(size=(n, n - p))
    G_orth = G - Qx @ (Qx.T @ G)
    # Orthonormalize to get Q_perp (n x (n-p))
    Q_perp, _ = np.linalg.qr(G_orth, mode="reduced")

    # --- We need an (n-p) x p matrix W with W^T W = M ---
    # Construct W = U0 @ sqrtM, where U0 has orthonormal columns: U0^T U0 = I_p.
    # This requires n - p >= p (checked above).
    G2 = rng.standard_normal(size=(n - p, p))
    # Orthonormalize columns to get U0 (n-p x p)
    U0, _ = np.linalg.qr(G2, mode="reduced")
    # It is possible that QR returns fewer columns if rank-deficient; guard that:
    if U0.shape != (n - p, p):
        raise RuntimeError("Failed to build a full p-column orthonormal U0 in R^{n-p}.")

    W = U0 @ sqrtM  # (n-p) x p
    # --- Assemble knockoffs: Xk = X A + Q_perp W  (n x p) ---
    X_knock = X @ A + Q_perp @ W
    if not np.all(np.isfinite(X_knock)):
        warnings.warn(
            "Non-finite entries detected in knockoff matrix; applying nan_to_num.",
            RuntimeWarning,
        )
        X_knock = np.nan_to_num(X_knock, nan=0.0, posinf=0.0, neginf=0.0)

    meta = dict(
        Sigma=Sigma_psd,
        s=float(s),
        lambda_min=float(lambda_min),
        S=S,
        A=A,
        M=M,
        n=int(n),
        p=int(p),
        rank_M=rank_M,
        seed=None if seed is None else int(seed),
    )

    return X_knock, meta


def lasso_path_stats(
    X: np.ndarray,
    y: np.ndarray,
    Xk: np.ndarray,
    *,
    coef_tol: float = 1e-9,
    max_iter: int = 10000,   # kept for symmetry; lars_path ignores it in some versions
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    Compute Barber–Candès lasso signed-max statistic from a SINGLE LARS path on [X | Xk].

    Precondition:
      - Columns of X already centered and variance=1 (same for Xk).
      - y centered (mean zero).
    Returns:
      dict(Z=..., Z_tilde=..., W=sign(Z - Zt) * max(Z, Zt))
    """
    assert X.shape == Xk.shape
    n, p = X.shape

    Xc  = X - X.mean(axis=0, keepdims=True)
    Xkc = Xk - Xk.mean(axis=0, keepdims=True)

    std_x = np.std(Xc, axis=0, ddof=1, keepdims=True)
    std_xk = np.std(Xkc, axis=0, ddof=1, keepdims=True)

    Xc = Xc / np.maximum(std_x, 1e-12)
    Xkc = Xkc / np.maximum(std_xk, 1e-12)

    y0  = y - y.mean()

    # OLD VERSION
    X_aug = np.hstack([Xc, Xkc])

    # NEW VERSION

    # X_aug = np.empty((n, 2 * p), dtype=float)
    # X_aug[:, :p] = Xc
    # X_aug[:, p:] = Xkc

    # Extra numeric safety
    if not np.all(np.isfinite(X_aug)) or not np.all(np.isfinite(y0)):
        warnings.warn(
            "Non-finite values in X_aug or y0; applying nan_to_num before LARS.",
            RuntimeWarning,
        )
        X_aug = np.nan_to_num(X_aug, nan=0.0, posinf=0.0, neginf=0.0)
        y0 = np.nan_to_num(y0, nan=0.0, posinf=0.0, neginf=0.0)

    # lars_path signature without normalize/fit_intercept (preprocessed data!)
    # method="lasso" gives the Lasso/LARS path with alphas on correlation scale
    alphas, _, coefs = lars_path(X_aug, y0, method="lasso", verbose=False)
    # coefs shape: (2p, n_alphas), alphas decreasing

    # First-entry alpha per feature
    nz = (np.abs(coefs) > coef_tol)
    # OLD VERSION
    # entry = np.zeros(2 * p)
    # for j in range(2 * p):
    #     idx = np.argmax(nz[j])
    #     entry[j] = alphas[idx] if nz[j].any() else 0.0

    # NEW VERSION
    has_nonzero = nz.any(axis=1)
    first_idx = nz.argmax(axis=1)
    entry = np.where(has_nonzero, alphas[first_idx], 0.0)

    Z  = entry[:p]
    Zt = entry[p:]
    W  = np.sign(Z - Zt) * np.maximum(Z, Zt)   # signed–max

    return {"Z": Z, "Z_tilde": Zt, "W": W}

####################################### helper functions to improve knockoff_threshold ################################
def _threshold_vectorized(W: np.ndarray, candidates: np.ndarray, q: float, offset: int) -> float:
    """Fully vectorized threshold computation (best for small p)."""
    W_col = W[:, np.newaxis]
    t_row = candidates[np.newaxis, :]

    num_pos = np.sum(W_col >= t_row, axis=0)
    num_neg = np.sum(W_col <= -t_row, axis=0)

    fdp_hat = (offset + num_neg) / np.maximum(1, num_pos)

    valid_idx = np.where(fdp_hat <= q)[0]

    if valid_idx.size == 0:
        return float("nan")

    return float(candidates[valid_idx[0]])

def _threshold_loop(W: np.ndarray, candidates: np.ndarray, q: float, offset: int) -> float:
    """Loop-based threshold computation (better memory for large p)."""
    for t in candidates:
        num = offset + np.count_nonzero(W <= -t)
        den = max(1, int(np.count_nonzero(W >= t)))
        if num / den <= q:
            return float(t)
    return float("nan")

#################################################################################################################
def knockoff_threshold(W: np.ndarray, q: float, offset: int = 1) -> float:
    """
    Compute the Knockoff / Knockoff+ data-dependent threshold.

    Definition (Barber & Candès, 2015):
        T = min{ t > 0 :
                 (offset + # { j : W_j <= -t }) / max(1, # { j : W_j >= t }) <= q }
    where:
      - offset = 1  -> Knockoff+ (exact FDR control, recommended)
      - offset = 0  -> original Knockoff (slightly more liberal)

    Parameters
    ----------
    W : array-like, shape (p,)
        Antisymmetric statistics (positive favors original, negative favors knockoff).
    q : float
        Target FDR level in (0, 1).
    offset : int, default 1
        Use 1 for Knockoff+. Use 0 for the original Knockoff.

    Returns
    -------
    T : float or np.nan
        The chosen threshold. np.nan if no t satisfies the inequality (i.e., select none).

    Notes
    -----
    - We search over the **data-driven grid** of positive unique |W_j| values.
    - If all W are zero or NaN/inf, returns np.nan.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 1:
        raise ValueError("W must be a 1D array.")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1).")
    if offset not in (0, 1):
        raise ValueError("offset must be 0 (Knockoff) or 1 (Knockoff+).")

    # Clean and build candidate grid t > 0 from data
    w_clean = W[np.isfinite(W)]
    w_abs = np.abs(w_clean[w_clean != 0.0])
    if w_abs.size == 0:
        return float("nan")

    candidates = np.sort(np.unique(w_abs))

    # Scan in increasing t to find the first that satisfies the inequality
    # (earliest t gives the *smallest* set of selections that achieves FDP_hat <= q)
    # OLD VERSION
    # for t in candidates:
    #     num = offset + np.count_nonzero(W <= -t)
    #     den = max(1, int(np.count_nonzero(W >= t)))
    #     fdp_hat = num / den
    #     if fdp_hat <= q:
    #         return float(t)

    # NEW VERSION
    p = len(W)
    n_candidates = len(candidates)

    # Adaptive algorithm selection based on problem size
    if p < 500 and n_candidates < 500:
        return _threshold_vectorized(W, candidates, q, offset)
    else:
        return _threshold_loop(W, candidates, q, offset)
    # No feasible t -> select none
    return float("nan")


def knockoff_select(W: np.ndarray, q: float, offset: int = 1) -> tuple[np.ndarray, dict]:
    """
    Select features using the Knockoff / Knockoff+ thresholding rule.

    Parameters
    ----------
    W : array-like, shape (p,)
        Antisymmetric Knockoff statistics.
    q : float
        Target FDR level in (0, 1).
    offset : int, default 1
        1 for Knockoff+ (recommended), 0 for original Knockoff.

    Returns
    -------
    selected : np.ndarray, shape (k,)
        Indices j such that W_j >= T (empty if no feasible threshold).
    info : dict
        {'T': threshold (float or np.nan), 'fdp_hat': float or np.nan}
    """

    W = np.asarray(W, float)
    # OLD VERSION
    # tgrid = np.unique(np.abs(W)[np.abs(W) > 0.0])
    # tgrid.sort()
    #
    # T = None;
    # fdp_hat_T = None
    # for t in tgrid:
    #     num = offset + np.sum(W <= -t)  # Knockoff+ numerator
    #     den = max(1, int(np.sum(W >= t)))  # denominator
    #     fdp_hat = num / den
    #     if fdp_hat <= q:
    #         T = float(t);
    #         fdp_hat_T = float(fdp_hat)
    #         break
    #
    # if T is None:
    #     return np.array([], dtype=int), {"T": None, "fdp_hat": None}
    # NEW VERSION
    T = knockoff_threshold(W, q, offset)

    if np.isnan(T):
        return np.array([], dtype=int), {"T": None, "fdp_hat": None}

    num = offset + np.sum(W <= -T)
    den = max(1, int(np.sum(W >= T)))
    fdp_hat = float(num / den)

    selected = np.where(W >= T)[0].astype(int)
    return selected, {"T": T, "fdp_hat": fdp_hat}



def bh_select_marginal(X: np.ndarray, y: np.ndarray, q: float = 0.2, by_correction: bool = False, eps: float = 1e-12):
    """
    Benjamini–Hochberg baseline using marginal Pearson correlations (two-sided).
    Optional Benjamini–Yekutieli log-factor correction via q -> q / S(p).

    Parameters
    ----------
    X : (n, p) array
        Design (need not be pre-centered/normalized).
    y : (n,) array
        Response.
    q : float in (0,1)
        Target FDR level.
    by_correction : bool
        If True, use BY correction: q_eff = q / sum_{i=1}^p 1/i.
    eps : float
        Numerical guard to avoid division by zero.

    Returns
    -------
    selected : set[int]
        Indices selected by BH.
    info : dict
        {'pvals', 'q_eff', 'threshold', 'm', 'k'}
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    n, p = X.shape

    if y.shape[0] != n:
        raise ValueError("Length of y must match number of rows in X.")
    if not (0 < q < 1):
        raise ValueError("q must be in (0, 1).")
    if n < 3:
        # Need at least 3 points for correlation t-test with df = n-2
        return set(), {"pvals": np.ones(p), "q_eff": q, "threshold": np.nan, "m": p, "k": 0}

    # Center
    yc = y - y.mean()
    Xc = X - X.mean(axis=0, keepdims=True)

    # Norms (equivalent to sd * sqrt(n-1)); safe against zeros
    sy = np.linalg.norm(yc)
    sx = np.linalg.norm(Xc, axis=0)

    safe_sy = max(sy, eps)
    safe_sx = np.where(sx < eps, np.inf, sx)  # columns with ~zero variance → r=0 later

    # Pearson correlations r_j = <Xc_j, yc> / (||Xc_j|| * ||yc||)
    r = (Xc.T @ yc) / (safe_sx * safe_sy)
    r = np.where(np.isfinite(r), r, 0.0)
    # Clip to open interval for stability in t-transform
    r = np.clip(r, -1 + 1e-15, 1 - 1e-15)

    # Two-sided p-values via t = r * sqrt(df / (1 - r^2)), df = n - 2
    df = n - 2
    t_stat = r * np.sqrt(df / np.maximum(1e-15, 1.0 - r * r))
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df))

    # BY/log-factor correction (optional)
    q_eff = q
    if by_correction:
        S_p = np.sum(1.0 / np.arange(1, p + 1, dtype=float))
        q_eff = min(q / S_p, 1.0)

    # BH step-up
    order = np.argsort(pvals)
    pv_sorted = pvals[order]
    thresh = q_eff * (np.arange(1, p + 1, dtype=float) / p)
    below = pv_sorted <= thresh

    if not np.any(below):
        return set(), {"pvals": pvals, "q_eff": q_eff, "threshold": np.nan, "m": p, "k": 0}

    k_max = int(np.max(np.where(below)[0])) + 1
    p_cutoff = float(pv_sorted[k_max - 1])
    selected = set(order[:k_max].tolist())

    info = dict(pvals=pvals, q_eff=q_eff, threshold=p_cutoff, m=p, k=len(selected))
    return selected, info


def bh_select_whitened(X, y, q=0.2, ridge=1e-8):
    """
    BH after decorrelating marginal z using Sigma^{-1/2}.
    Assumes columns of X are normalized and noise var=1 (your DGP).
    """
    n, p = X.shape
    z = X.T @ y                     # marginal z (unnormalized but proportional)

    Sigma = (X.T @ X) / float(n)    # p x p; diag ≈ 1 if columns normalized

    # Robust inverse sqrt via eigen-decomposition with a tiny ridge
    w, V = np.linalg.eigh(Sigma + ridge * np.eye(p))
    w_inv_sqrt = 1.0 / np.sqrt(np.maximum(w, 1e-12))
    Sigma_inv_sqrt = (V * w_inv_sqrt) @ V.T
    z_whiten = Sigma_inv_sqrt @ z

    pvals = 2.0 * (1.0 - norm.cdf(np.abs(z_whiten)))

    order = np.argsort(pvals)
    pv_sorted = pvals[order]
    thresh = q * (np.arange(1, p + 1) / p)


    # OLD VERSION
    # k = np.where(pv_sorted <= thresh)[0].max() + 1 if np.any(pv_sorted <= thresh) else 0
    # sel_idx = set(order[:k].tolist())
    # return sel_idx, {"k": k}

    #NEW VERSION
    valid_idx = np.where(pv_sorted <= thresh)[0]
    k = int(valid_idx.max()) + 1 if valid_idx.size > 0 else 0

    return set(order[:k].tolist()), {"k": k}




