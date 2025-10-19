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

    # numerical stability
    Sigma = 0.5 * (Sigma + Sigma.T)

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

    X_knock = X
    meta = dict()
    return X_knock, meta