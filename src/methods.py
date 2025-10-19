import numpy as np
from sklearn.linear_model import lasso_path
from scipy import stats

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
        Sigma = np.asarray(use_true_Sigma, dtype=np.float64)
        if Sigma.shape != (p, p):
            raise ValueError("use_true_Sigma must have shape (p, p).")

    # Symmetrize
    Sigma = 0.5 * (Sigma + Sigma.T)

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
    center_y: bool = True,
    n_alphas: int = 100,
    eps: float = 1e-3,
    coef_tol: float = 1e-9,
    max_iter: int = 10_000,
) -> dict:
    """
    Compute Knockoff W-statistics via a single Lasso path on the augmented design [X | Xk].

    For each original variable j and its knockoff j~, define the entry alpha:
        Z_j      = the largest alpha where coef_j first becomes nonzero (as alphas decrease)
        Z_j_tilde = same for the knockoff column
    Then the antisymmetric statistic is:
        W_j = max(Z_j, Z_j_tilde) * sign(Z_j - Z_j_tilde)

    Notes
    -----
    - We use sklearn.linear_model.lasso_path (alphas returned in decreasing order).
    - Columns of X are assumed to be on a common scale already (e.g., ||X[:,j]||_2 = sqrt(n)).
    - If a feature never becomes nonzero along the path, its Z is set to 0.

    Parameters
    ----------
    X : array-like, shape (n, p)
        Original design.
    y : array-like, shape (n,)
        Response. Optionally centered internally.
    Xk : array-like, shape (n, p)
        Knockoff design.
    center_y : bool, default True
        If True, center y to mean zero before fitting (keeps fit_intercept=False stable).
    fit_intercept : bool, default False
        Passed to lasso_path. (When center_y=True, leaving this False is typical.)
    n_alphas : int, default 100
        Number of alphas on the path (sklearn will auto-generate when alphas=None).
    eps : float, default 1e-3
        Length of the path (ratio alpha_min / alpha_max).
    coef_tol : float, default 1e-9
        Threshold for treating a coefficient as nonzero.
    max_iter : int, default 10_000
        Max iterations per coordinate descent subproblem.

    Returns
    -------
    out : dict with keys
        - 'W'        : (p,) array, antisymmetric Knockoff statistics
        - 'Z_orig'   : (p,) entry alphas for original columns
        - 'Z_knock'  : (p,) entry alphas for knockoff columns
        - 'alphas'   : (n_alphas,) decreasing sequence used by sklearn
        - 'coefs'    : (2p, n_alphas) coefficient path for [X | Xk]
        - 'coef_tol' : float, tolerance used to decide 'nonzero'
    """
    X = np.asarray(X, dtype=np.float64, order="C")
    Xk = np.asarray(Xk, dtype=np.float64, order="C")
    y = np.asarray(y, dtype=np.float64).ravel()

    n, p = X.shape
    if Xk.shape != (n, p):
        raise ValueError(f"Xk must have shape {(n, p)}, got {Xk.shape}.")
    if y.shape[0] != n:
        raise ValueError(f"y must have length n={n}, got {y.shape[0]}.")

    # Augment design
    X_aug = np.concatenate([X, Xk], axis=1)

    # Optionally center y (keep fit_intercept=False for a pure path on X_aug)
    y_fit = y - y.mean() if center_y else y

    # Run a single joint Lasso path
    # Returns: alphas (decreasing), coefs (n_features x n_alphas)
    # We pass max_iter via **kwargs to ensure convergence on tougher problems.
    alphas, coefs, _ = lasso_path(
        X_aug,
        y_fit,
        eps=eps,
        n_alphas=n_alphas,
        alphas=None,
        max_iter=max_iter,
        verbose=False,
    )

    # Ensure shapes are consistent with sklearn’s contract
    # coefs has shape (2p, n_alphas); alphas shape (n_alphas,)
    if coefs.shape != (2 * p, alphas.shape[0]):
        raise RuntimeError("Unexpected shapes from lasso_path: "
                           f"coefs {coefs.shape}, alphas {alphas.shape}")

    # Find entry alpha for each feature (first nonzero as alphas decrease)
    # Since alphas are decreasing, the FIRST index where |coef| > tol is the entry.
    Z_all = np.zeros(2 * p, dtype=np.float64)
    abs_coefs = np.abs(coefs)
    nz_mask = abs_coefs > coef_tol

    for j in range(2 * p):
        mask = nz_mask[j]
        if np.any(mask):
            first_idx = int(np.argmax(mask))  # first True along decreasing alphas
            Z_all[j] = float(alphas[first_idx])
        else:
            Z_all[j] = 0.0

    Z_orig = Z_all[:p]
    Z_knock = Z_all[p:]

    # Antisymmetric W statistics
    # If both Z's are zero, W_j = 0.
    max_pair = np.maximum(Z_orig, Z_knock)
    diff = Z_orig - Z_knock
    W = max_pair * np.sign(diff)

    out = dict(
        W=W,
        Z_orig=Z_orig,
        Z_knock=Z_knock,
        alphas=alphas,
        coefs=coefs,
        coef_tol=float(coef_tol),
    )
    return out


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
    candidates = np.sort(np.unique(np.abs(w_clean[w_clean != 0.0])))
    if candidates.size == 0:
        return float("nan")

    # Scan in increasing t to find the first that satisfies the inequality
    # (earliest t gives the *smallest* set of selections that achieves FDP_hat <= q)
    for t in candidates:
        num = offset + np.count_nonzero(W <= -t)
        den = max(1, int(np.count_nonzero(W >= t)))
        fdp_hat = num / den
        if fdp_hat <= q:
            return float(t)

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
        {'T': threshold (float or np.nan), 'fdp_hat': float or np.nan,
         'num_neg': int, 'num_pos': int}
    """
    T = knockoff_threshold(W, q=q, offset=offset)
    if not np.isfinite(T):
        return np.array([], dtype=int), {
            "T": float("nan"),
            "fdp_hat": float("nan"),
            "num_neg": int(np.count_nonzero(W < 0)),
            "num_pos": int(np.count_nonzero(W > 0)),
        }

    sel_mask = (W >= T)
    selected = np.flatnonzero(sel_mask)

    num = offset + int(np.count_nonzero(W <= -T))
    den = max(1, int(sel_mask.sum()))
    fdp_hat = num / den

    return selected.astype(int), {
        "T": float(T),
        "fdp_hat": float(fdp_hat),
        "num_neg": int(np.count_nonzero(W <= -T)),
        "num_pos": int(np.count_nonzero(W >= T)),
    }


def bh_select_marginal(X: np.ndarray, y: np.ndarray, q: float = 0.2) -> tuple[np.ndarray, dict]:
    """
    Benjamini–Hochberg baseline using marginal correlations.

    Parameters
    ----------
    X : (n, p) ndarray
        Design matrix (columns normalized or not; normalization handled internally).
    y : (n,) ndarray
        Response vector.
    q : float, default=0.2
        Target FDR level.

    Returns
    -------
    selected : (k,) ndarray of ints
        Indices of selected features after BH correction.
    info : dict
        {'pvals': p-values array, 'q': q, 'threshold': p-value cutoff, 'm': p, 'k': number selected}
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("Length of y must match number of rows in X.")
    if not (0 < q < 1):
        raise ValueError("q must be in (0, 1).")

    # Center and normalize y
    y_centered = y - y.mean()
    y_std = np.linalg.norm(y_centered)
    if y_std < 1e-12:
        raise ValueError("y has zero variance.")
    y_norm = y_centered / y_std

    # Marginal correlations
    X_centered = X - X.mean(axis=0, keepdims=True)
    X_std = np.linalg.norm(X_centered, axis=0)
    X_std = np.where(X_std < 1e-12, 1.0, X_std)
    X_norm = X_centered / X_std
    r = X_norm.T @ y_norm  # correlation-like scores

    # Convert to two-sided p-values (using t-distribution under H0)
    # df = n - 2, t = r * sqrt(df / (1 - r^2))
    df = max(n - 2, 1)
    t_stat = r * np.sqrt(df / np.maximum(1e-12, 1 - r**2))
    pvals = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

    # BH procedure
    m = p
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    thresh = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresh
    if not np.any(below):
        return np.array([], dtype=int), {'pvals': pvals, 'q': q, 'threshold': np.nan, 'm': m, 'k': 0}

    k_max = np.max(np.where(below)[0]) + 1
    p_cutoff = sorted_p[k_max - 1]
    selected = np.flatnonzero(pvals <= p_cutoff)

    info = dict(
        pvals=pvals,
        q=q,
        threshold=p_cutoff,
        m=m,
        k=len(selected),
    )
    return selected.astype(int), info


