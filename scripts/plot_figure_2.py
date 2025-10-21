import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lars_path
from numpy.linalg import cholesky
from pathlib import Path
import argparse

def first_entry_lambdas_lars(X, y):
    # X: columns already centered & std=1; y centered
    alphas, _, coefs = lars_path(X, y, method="lar")
    nz = (np.abs(coefs) > 0)
    entry = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        entry[j] = alphas[np.argmax(nz[j])] if nz[j].any() else 0.0
    return entry


def simulate_figure2_paper(n=300, p=100, k=30, rho=0.3, seed=1):
    """
    Paper-spec simulator:
      - Rows of X ~ N(0, Sigma) with Sigma_ii=1 and Sigma_ij=rho (i!=j)
      - Center & standardize columns of X
      - y = 3.5 * sum_{j=1..k} X_j + z,  z ~ N(0, I_n)
    """
    rng = np.random.default_rng(seed)
    # Equicorrelated covariance: Sigma = (1-rho) I + rho * 11^T
    # Draw X via eigen decomposition for numerical stability
    Sigma = (1 - rho) * np.eye(p) + rho * np.ones((p, p))
    # Sample multivariate normal rows
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
    # Center and standardize columns
    X = X - X.mean(axis=0, keepdims=True)
    X = X / X.std(axis=0, ddof=1, keepdims=True)

    # Signals: first k columns
    y = 3.5 * X[:, :k].sum(axis=1) + rng.standard_normal(n)
    y = y - y.mean()  # no intercept in LARS call

    sig_idx = np.arange(k)
    null_idx = np.arange(k, p)
    return X, y, sig_idx, null_idx


def permute_columns(X, rng):
    Xtil = X.copy()
    n, p = X.shape
    for j in range(p):
        Xtil[:, j] = Xtil[rng.permutation(n), j]
    return Xtil


def ecdf(x):
    x = np.sort(x)
    y = np.linspace(0, 1, len(x), endpoint=True)
    return x, y

def plot_figure2_bar(Z, Zt, sig_idx, null_idx, outpath=None,
                     title="Lasso path entry (Figure 2 style)"):
    """
    Reproduce Figure 2: Value of lambda when each variable enters the Lasso path.
    Z : entry lambda for original features
    Zt: entry lambda for permuted features
    sig_idx, null_idx: indices of signal and null features in Z
    """
    p = Z.shape[0]
    fig, ax = plt.subplots(figsize=(10, 3.2))

    # x-positions for originals and permuted columns
    x_orig = np.arange(1, p + 1)
    x_perm = np.arange(p + 1, 2 * p + 1)

    # plot originals: black squares = non-nulls, red circles = nulls
    ax.scatter(x_orig[sig_idx], Z[sig_idx],
               s=25, c='black', marker='s', label='Original non–null features')
    ax.scatter(x_orig[null_idx], Z[null_idx],
               s=20, c='red', alpha=0.8, label='Original null features')

    # plot permuted features
    ax.scatter(x_perm, Zt, s=30, c='blue', marker='^', label='Permuted features')

    # axes, labels, and limits
    ax.set_xlim(0, 2 * p + 5)
    ax.set_ylim(0, 1.05 * max(Z.max(), Zt.max()))
    ax.set_xlabel(r"Index of column in the augmented matrix $[X \; X^{\pi}]$")
    ax.set_ylabel(r"Value of $\lambda$ when variable enters model")
    ax.legend(frameon=True, loc='upper right')
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150)
    return fig, ax


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Figure 2 (paper spec)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="artifacts/knockoffs/figure2_lasso_path.png")
    args = ap.parse_args()

    X, y, sig_idx, null_idx = simulate_figure2_paper(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    Xperm = permute_columns(X, rng)

    # Augmented design [X | X^pi]; columns already standardized
    X_aug = np.hstack([X, Xperm])

    entry = first_entry_lambdas_lars(X_aug, y)
    p = X.shape[1]
    Z  = entry[:p]      # originals
    Zt = entry[p:]      # permuted

    title = "Figure 2 (paper spec): original nulls enter earlier than permuted"
    plot_figure2_bar(Z, Zt, sig_idx, null_idx, outpath=args.out, title=title)
    print(f"Saved: {args.out}")

    corr = np.corrcoef(X, rowvar=False)
    m1 = np.mean(np.abs(corr[null_idx[:,None], sig_idx].reshape(len(null_idx), -1)))
    # permuted columns are independent across rows, so approximate corr via sample on Xperm
    corr_perm = np.corrcoef(np.hstack([X[:, sig_idx], Xperm[:, null_idx]]), rowvar=False)
    m2 = np.mean(np.abs(corr_perm[:len(sig_idx), len(sig_idx):]))
    print(f"Avg |corr(null originals, signals)| ≈ {m1:.3f}")
    print(f"Avg |corr(permuted, signals)|      ≈ {m2:.3f}")

if __name__ == "__main__":
    main()
