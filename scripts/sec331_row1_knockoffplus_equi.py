import argparse
import math

import pandas as pd
import numpy as np
from tqdm import trange
from pathlib import Path
from typing import Iterable, Sequence, Union, Dict, Any
from src.metrics import fdp_power, fdr_power_all
from src.dgps import generate_full
from src.methods import knockoffs_equicorr, lasso_path_stats, knockoff_select
from src.simulation import _trial_seeds

"""
Section 3.3.1 — Row 1: Knockoff+ (equivariant construction)
n=3000, p=1000, k=30, beta in {±3.5}, X_ij ~ N(0,1) then column-normalized, y ~ N(Xβ, I).
Requires: knockpy  (pip install knockpy)
"""

def knockoff_select_plus(W, q):
    W = np.asarray(W, dtype=float)
    ts = np.sort(np.unique(np.abs(W)))
    selected = np.array([], dtype=int)
    for t in ts:
        num = np.sum(W <= -t)
        den = np.sum(W >=  t)
        if den == 0:
            continue
        if (1 + num) / den <= q:
            selected = np.where(W >= t)[0]
            break
    return selected


def run_one_trial_row1(cfg: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
    """
    §3.3.1 Row 1 only: Knockoff+ with equicorrelated construction.
    Returns a flat dict (per-trial) with FDP/Power for Knockoff+.
    """
    required_syms = [
        ("generate_full", generate_full),
        ("knockoffs_equicorr", knockoffs_equicorr),
        ("lasso_path_stats", lasso_path_stats),
        ("knockoff_select", knockoff_select),
        ("fdp_power", fdp_power),
    ]
    missing = [name for name, sym in required_syms if sym is None]
    if missing:
        raise RuntimeError(f"Missing required symbols: {missing}")

    # 1) Data (paper spec: iid N(0,1), columns normalized to var=1, k=30, A=3.5)
    ds_seed = int(cfg["seed"]) + int(trial_id)
    y, X, beta, meta = generate_full(
        int(cfg["n"]), int(cfg["p"]),
        mode="iid",                 # §3.3.1 uses iid N(0,1)
        rho=None, df=math.inf,
        k=int(cfg["k"]), A=float(cfg["A"]),
        normalize=True,             # normalize columns…
        norm_target="unit_var",     # …to variance 1 (NOT sqrt_n)
        seed=ds_seed,
    )
    true_support = meta["support_indices"]

    # 2) Knockoffs: equicorrelated construction
    seed_kn, seed_lasso = _trial_seeds(int(cfg["seed"]), int(trial_id))
    Xk, info_k = knockoffs_equicorr(X, seed=seed_kn)  # uses sample Sigma of X

    corr = np.corrcoef(np.hstack([X, Xk]), rowvar=False)
    p = X.shape[1]
    # sample cross-covariances (should be ≈ Sigma - S)
    print("mean diag corr(X, Xk):", np.mean([corr[j, p + j] for j in range(p)]))
    print("mean offdiag corr(X, Xk):", np.mean(corr[:p, p:]))

    # 3) W statistics from a single lasso path over [X | X~]
    #    Your lasso_path_stats should:
    #      - center y inside or use y as given (already mean-zero from generate_full if you coded it so);
    #      - NOT re-normalize columns (X already var=1; X~ should mimic X's scale);
    #      - return dict with 'W' equal to signed-max (preferred) or difference; we want signed-max.
    stats = lasso_path_stats(
        X, y, Xk,
        n_alphas=int(cfg.get("n_alphas", 200)),
        eps=float(cfg.get("eps", 1e-3)),
        coef_tol=float(cfg.get("coef_tol", 1e-9)),
        max_iter=int(cfg.get("max_iter", 10000)),
        # If your implementation exposes these, make sure they are False:
        # fit_intercept=False, normalize=False
    )
    W = stats["W"]

    # After computing W
    W = stats["W"]
    S = set(map(int, true_support))
    W_sig = np.asarray([W[j] for j in S])
    W_null = np.asarray([W[j] for j in range(len(W)) if j not in S])

    print("median |W_null|:", np.median(np.abs(W_null)))
    print("median |W_sig| :", np.median(np.abs(W_sig)))

    print(
        f"W>0 frac={np.mean(W > 0):.3f}, W<0 frac={np.mean(W < 0):.3f}, "
        f"med|W|={np.median(np.abs(W)):.3f}, p95|W|={np.quantile(np.abs(W), 0.95):.3f}"
    )
    print(
        f"signals: minW={W_sig.min():.3f}, q25={np.quantile(W_sig, 0.25):.3f}, "
        f"q50={np.median(W_sig):.3f}, q75={np.quantile(W_sig, 0.75):.3f}"
    )
    print(
        f"nulls  : minW={W_null.min():.3f}, q25={np.quantile(W_null, 0.25):.3f}, "
        f"q50={np.median(W_null):.3f}, q75={np.quantile(W_null, 0.75):.3f}"
    )
    # 4) Knockoff+ selection at q
    sel_kn  = knockoff_select_plus(W, q=float(cfg["q"]))  # Knockoff+
    sel_kn = np.asarray(sel_kn, dtype=int)

    # sanity print
    TP = int(np.intersect1d(sel_kn, true_support).size)
    #print(
        #f"T={info_sel.get('T', None)}, FDPhat={info_sel.get('fdp_hat', None)}, "
       # f"R={len(sel_kn)}, TP={TP}, minW_signal={W_sig.min():.4f}"
   # )
    print(type(true_support), getattr(true_support, "dtype", None), len(true_support))
    print(type(sel_kn), getattr(sel_kn, "dtype", None), len(sel_kn))

    # 5) Per-trial metrics (Row 1 only)
    m_kn = fdp_power(true_support, sel_kn)

    return dict(
        name=cfg["name"],
        trial=int(trial_id),
        n=int(cfg["n"]), p=int(cfg["p"]),
        k_true=int(cfg["k"]), A=float(cfg["A"]),
        method="Knockoff+ (equi)",
        R_kn=m_kn["R"],
        FDP_kn=m_kn["FDP"], Power_kn=m_kn["Power"],
        #T_kn=info_sel.get("T", None),
        w_nonzero=int(np.sum(W != 0.0)),
        seed_dataset=ds_seed,
        seed_knockoff=seed_kn,
    )

def simulate_one(n=3000, p=1000, k=30, A=3.5, rng=None):
    """Return X (n×p, col-normalized), y, true_support (indices)."""
    rng = np.random.default_rng(rng)
    X = rng.standard_normal(size=(n, p))
    # column-normalize (mean 0, variance 1)
    X -= X.mean(axis=0, keepdims=True)
    X /= X.std(axis=0, ddof=1, keepdims=True)

    # choose support & signs
    support = rng.choice(p, size=k, replace=False)
    signs = rng.choice([-1.0, 1.0], size=k)
    beta = np.zeros(p)
    beta[support] = A * signs

    # response
    y = X @ beta + rng.standard_normal(n)
    y -= y.mean()  # no intercept in LARS

    return X, y, np.sort(support)

def run_simulation(cfg: Dict[str, Any]) -> pd.DataFrame:
    try:
        from tqdm import trange
    except Exception:
        trange = range
    rows = []
    for t in trange(int(cfg["n_trials"]), desc=f"Sim {cfg['name']}"):
        rows.append(run_one_trial_row1(cfg, t))   # <-- use the Row-1 variant here
    return pd.DataFrame(rows)

def summarize_row1(df):
    fdr = df["FDP_kn"].mean()
    power = df["Power_kn"].mean()
    T = len(df)
    fdr_se = df["FDP_kn"].std(ddof=1)/np.sqrt(T) if T>1 else np.nan
    power_se = df["Power_kn"].std(ddof=1)/np.sqrt(T) if T>1 else np.nan
    print("\n=== §3.3.1 Row 1: Knockoff+ (equicorrelated) ===")
    print(f"Trials: {T}")
    print(f"FDR   : {100*fdr:.2f}%  (SE {100*fdr_se:.2f}%)")
    print(f"Power : {100*power:.2f}%  (SE {100*power_se:.2f}%)")
    print("Paper target: FDR ≈ 14.40%, Power ≈ 60.99%")


def main():
    ap = argparse.ArgumentParser(
        description="§3.3.1 Row 1: Knockoff+ (equicorrelated) — replicate FDR/Power"
    )
    # Paper defaults
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--p", type=int, default=1000)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--A", type=float, default=3.5)
    ap.add_argument("--q", type=float, default=0.20)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="", help="Optional CSV path for per-trial summary.")
    args = ap.parse_args()

    # --- config dictionary used by run_one_trial_row1 ---
    cfg_row1 = dict(
        name="sec331_row1_knockoffplus_equi",
        n=3000,
        p=1000,
        k=30,
        A=3.5,
        q=0.20,
        n_trials=100,
        seed=1,
        # lasso path hyperparams (safe defaults)
        n_alphas=200,
        eps=1e-3,
        coef_tol=1e-9,
        max_iter=10000,
    )

    rows = []
    for t in trange(args.trials, desc="Knockoff+ (equi)"):
        rows.append(run_one_trial_row1(cfg_row1, t))
    df = pd.DataFrame(rows)

    # --- aggregate FDR and Power ---
    out = dict(
        FDR=df["FDP_kn"].mean(),
        Power=df["Power_kn"].mean(),
        FDR_se=df["FDP_kn"].std(ddof=1)/np.sqrt(len(df)),
        Power_se=df["Power_kn"].std(ddof=1)/np.sqrt(len(df)),
        num_trials=len(df),
    )

    print("\n=== §3.3.1 Row 1: Knockoff+ (equicorrelated) ===")
    print(f"Trials: {out['num_trials']}")
    print(f"FDR   : {100*out['FDR']:.2f}%  (SE {100*out['FDR_se']:.2f}%)")
    print(f"Power : {100*out['Power']:.2f}%  (SE {100*out['Power_se']:.2f}%)")
    print("Target (paper, Row 1): FDR ≈ 14.40%, Power ≈ 60.99%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Saved per-trial results to {args.out}")


if __name__ == "__main__": main()