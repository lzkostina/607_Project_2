import numpy as np

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