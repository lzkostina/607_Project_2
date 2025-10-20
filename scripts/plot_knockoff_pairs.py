import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Sequence, Union, Literal

ArrayLike = Union[np.ndarray, Sequence[int], Sequence[bool]]

# ---------- synthetic data ----------
def synth_knockoff_pairs(p=200, k=20, rho=0.3, base_mu=0.8, base_sigma=0.6,
                         signal_boost=1.0, seed: Optional[int]=None):
    """
    Create (Z, Z_tilde) with nulls ~ symmetric around diagonal and signals with Z > Z_tilde.
    All values are nonnegative to mimic 'entry' lambdas from a lasso path.
    """
    rng = np.random.default_rng(seed)
    S = rng.choice(p, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    Z  = np.zeros(p, dtype=float)
    Zt = np.zeros(p, dtype=float)

    # correlated noise
    L = np.array([[1.0, rho],[rho, 1.0]])
    C = np.linalg.cholesky(L)

    # Nulls
    null_idx = np.setdiff1d(np.arange(p), S)
    n0 = len(null_idx)
    if n0 > 0:
        base = rng.normal(base_mu, base_sigma, size=(n0, 1))
        eps = rng.normal(0, 0.5, size=(n0, 2)) @ C.T
        pair = np.clip(base + eps, a_min=0.0, a_max=None)
        Z[null_idx]  = pair[:,0]
        Zt[null_idx] = pair[:,1]
        # random swap for exchangeability look
        swap = rng.random(n0) < 0.5
        Z[null_idx][swap], Zt[null_idx][swap] = Zt[null_idx][swap].copy(), Z[null_idx][swap].copy()

    # Signals: push Z up relative to Z_tilde
    if k > 0:
        base = rng.normal(base_mu + signal_boost, base_sigma, size=(k, 1))
        epsZ  = rng.normal(0, 0.4, size=k)
        epsKt = rng.normal(0, 0.4, size=k)
        Z[S]  = np.clip(base.ravel() + 0.8*signal_boost + epsZ,  a_min=0.0, a_max=None)
        Zt[S] = np.clip(base.ravel() - 0.6*signal_boost + epsKt, a_min=0.0, a_max=None)

    return Z, Zt, S

# ---------- plotting ----------
def _to_signal_mask(true_support: Optional[ArrayLike], p: int) -> np.ndarray:
    if true_support is None:
        return np.zeros(p, dtype=bool)
    ts = np.asarray(true_support)
    if ts.dtype == bool:
        assert ts.size == p, "Boolean true_support mask must have length p."
        return ts
    idx = np.asarray(true_support, dtype=int)
    m = np.zeros(p, dtype=bool)
    m[idx] = True
    return m

def plot_knockoff_pairs(
    Z: np.ndarray,
    Z_tilde: np.ndarray,
    true_support: Optional[ArrayLike] = None,
    t: float = 1.5,
    stat: Literal["signed_max","diff"] = "signed_max",
    title: Optional[str] = None,
    savepath: Optional[Path] = None,
    dpi: int = 150
) -> dict:
    Z = np.asarray(Z, float).ravel()
    Zt = np.asarray(Z_tilde, float).ravel()
    assert Z.shape == Zt.shape, "Z and Z_tilde must have the same shape."
    p = Z.size

    is_sig = _to_signal_mask(true_support, p)
    is_null = ~is_sig

    lo = min(Z.min(), Zt.min()); hi = max(Z.max(), Zt.max())
    pad = 0.05 * (hi - lo + 1e-9)
    xmin, xmax = lo - pad, hi + pad
    ymin, ymax = xmin, xmax

    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    xs = np.linspace(xmin, xmax, 256)
    ax.plot(xs, xs, 'k-', lw=1)                    # diagonal

    if stat == "diff":
        # W = Z - Z~
        denom_mask = (Z - Zt) >= t           # below y = x - t
        num_mask   = (Z - Zt) <= -t          # above y = x + t
        ax.plot(xs, xs - t, 'k--', lw=1)
        ax.plot(xs, xs + t, 'k--', lw=1)
        ax.fill_between(xs, ymin, xs - t, alpha=0.10, color="gray")
        ax.fill_between(xs, xs + t, ymax, alpha=0.10, color="gray")
        W = Z - Zt
    else:
        # signed_max: W = sign(Z-Z~) * max(Z, Z~)
        below = Z > Zt
        above = Z < Zt
        denom_mask = below & (Z >= t)        # selected at threshold t
        num_mask   = above & (Zt >= t)       # negative side
        ax.axvline(t, ls='--', c='k', lw=1)
        ax.axhline(t, ls='--', c='k', lw=1)
        ax.fill_between([t, xmax], [ymin, ymin], [t, xmax], color="lightgray", alpha=0.6)
        ax.fill_betweenx([t, ymax], [ymin, ymin], [t, ymax], color="lightgray", alpha=0.6)
        W = np.sign(Z - Zt) * np.maximum(Z, Zt)

    # points
    ax.scatter(Z[is_null], Zt[is_null], c='k', s=18, alpha=0.75, label="Null features")
    if is_sig.any():
        ax.scatter(Z[is_sig],  Zt[is_sig],  c='red', marker='s', s=30, alpha=0.9, label="Non-null features")

    num_sel = int(denom_mask.sum()); num_neg = int(num_mask.sum())
    ax.text(0.02, 0.97, f"Selected (W≥t): {num_sel}\nNeg side (W≤−t): {num_neg}",
            transform=ax.transAxes, ha='left', va='top')

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect('equal', 'box')
    ax.set_xlabel(r"Value of $\lambda$ when $X_j$ enters model")
    ax.set_ylabel(r"Value of $\lambda$ when $\tilde X_j$ enters model")
    ax.legend(frameon=True, loc="upper right")
    ax.set_title(title or f"Estimated FDP at threshold t={t}")
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi)
    return {"num_sel": num_sel, "num_neg": num_neg, "W": W}

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Generate a Figure-1 style knockoff plot (synthetic or from CSV).")
    # synthetic controls
    ap.add_argument("--p", type=int, default=160, help="Total number of features.")
    ap.add_argument("--k", type=int, default=18, help="Number of non-null features.")
    ap.add_argument("--rho", type=float, default=0.25, help="Visual correlation of null pairs.")
    ap.add_argument("--base-mu", type=float, default=0.8, help="Baseline mean for entry lambdas.")
    ap.add_argument("--base-sigma", type=float, default=0.6, help="Baseline std for entry lambdas.")
    ap.add_argument("--signal-boost", type=float, default=1.2, help="How strongly signals fall below diagonal.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    # plotting/stat
    ap.add_argument("--t", type=float, default=1.5, help="Threshold t.")
    ap.add_argument("--stat", choices=["signed_max","diff"], default="signed_max",
                    help="Definition of W for shading/counts.")
    ap.add_argument("--title", type=str, default="FDR CONTROL VIA KNOCKOFFS (synthetic)", help="Plot title.")
    ap.add_argument("--out", type=Path, default=Path("artifacts/knockoffs/fig1_synthetic.png"),
                    help="Path to save PNG.")
    ap.add_argument("--save-csv", type=Path, default=None,
                    help="Optional path to save a CSV with columns: Z, Z_tilde, is_signal.")
    # optional: load Z/Z_tilde from CSV (columns Z,Z_tilde[,is_signal])
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="If provided, load Z and Z_tilde from this CSV instead of synthesizing.")
    args = ap.parse_args()

    if args.from_csv is not None:
        import pandas as pd
        df = pd.read_csv(args.from_csv)
        assert {"Z","Z_tilde"}.issubset(df.columns), "CSV must have columns 'Z' and 'Z_tilde'."
        Z, Zt = df["Z"].to_numpy(float), df["Z_tilde"].to_numpy(float)
        S = np.where(df.get("is_signal", np.zeros(len(df), dtype=bool)).astype(bool))[0]
    else:
        Z, Zt, S = synth_knockoff_pairs(
            p=args.p, k=args.k, rho=args.rho,
            base_mu=args.base_mu, base_sigma=args.base_sigma,
            signal_boost=args.signal_boost, seed=args.seed
        )

    # optional save of synthetic data
    if args.save_csv is not None:
        import pandas as pd
        outdf = pd.DataFrame({
            "Z": Z, "Z_tilde": Zt,
            "is_signal": np.isin(np.arange(len(Z)), S).astype(int)
        })
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        outdf.to_csv(args.save_csv, index=False)

    res = plot_knockoff_pairs(Z, Zt, true_support=S, t=args.t,
                              stat=args.stat, title=args.title, savepath=args.out)
    print(f"Saved plot to: {args.out}")
    print(f"Counts: selected (W>=t) = {res['num_sel']}, negative side (W<=-t) = {res['num_neg']}")

if __name__ == "__main__":
    main()
