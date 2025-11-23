from __future__ import annotations

import numpy as np
import pandas as pd
import argparse
import json
import math
import sys
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from functools import partial

# Try to import joblib for parallelization
try:
    from joblib import Parallel, delayed, cpu_count
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("joblib not available; falling back to sequential execution")


# ----------------------------- Paths & IO helpers -----------------------------

RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw"
FIG_DIR = RESULTS_DIR / "figures"
for _d in (RAW_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------- Defaults -----------------------------------

DEFAULTS: Dict[str, Any] = {
    # DGP
    "mode": "iid",          # {"iid","ar1"}
    "rho": None,            # required iff mode == "ar1"
    "k": 30,
    "A": 3.5,
    "df": math.inf,         # Gaussian noise
    "normalize": True,
    "norm_target": "sqrt_n",

    # Knockoff+/BH (placeholders until we wire methods)
    "q": 0.2,
    "n_alphas": 150,
    "eps": 1e-3,
    "coef_tol": 1e-9,
    "max_iter": 10_000,

    # Trials/reproducibility
    "n_trials": 100,
    "seed": 12345,
    # Parallelization options
    "n_jobs": -1,
    "batch_size": "auto",
    "backend": "loky",
    # NEW: Optimization options
    "use_true_sigma": True,  # Use theoretical Sigma for knockoffs (faster for AR(1))
    "precompute_sigma": True,  # Precompute Sigma before trials
    "vectorize_metrics": True,  # Use vectorized metric computation
}


def _coerce_df(df_val):
    """Return a numeric df with math.inf allowed; raise if invalid."""
    if df_val is None:
        return math.inf
    if isinstance(df_val, (int, float)):
        # accept positive numbers or inf
        if df_val > 0 or math.isinf(df_val):
            return float(df_val)
        raise ValueError("df must be positive or math.inf.")
    if isinstance(df_val, str):
        s = df_val.strip().lower()
        if s in {"inf", "+inf", "infinity", "+infinity"}:
            return math.inf
        # allow numeric strings like "5" or "5.0"
        try:
            v = float(s)
        except ValueError:
            raise ValueError(f"Could not parse df='{df_val}' as a number or infinity.")
        if v <= 0:
            raise ValueError("df must be positive or math.inf.")
        return v
    raise TypeError(f"Unsupported type for df: {type(df_val)}")


def load_config(path: Path) -> Dict[str, Any]:
    user = read_json(path)
    for key in ("name", "n", "p"):
        if key not in user:
            raise KeyError(f"Config {path} missing required key: {key!r}")
    cfg = {**DEFAULTS, **user}
    # NEW: normalize df
    cfg["df"] = _coerce_df(cfg.get("df", DEFAULTS.get("df", math.inf)))
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Fail-fast checks with actionable hints.
    """
    mode = cfg["mode"]
    if mode not in {"iid", "ar1"}:
        raise ValueError(f"mode must be 'iid' or 'ar1' (got {mode!r}).")
    if mode == "ar1":
        rho = cfg.get("rho", None)
        if rho is None:
            raise ValueError("For mode='ar1', set rho (e.g., 0.5).")
        if not (-0.999 < float(rho) < 0.999):
            raise ValueError("rho must be in (-0.999, 0.999) for numerical stability.")
    df = cfg.get("df", math.inf)
    if not (df > 0 or math.isinf(df)):
        raise ValueError("df must be positive or math.inf.")
    n, p, k, q = int(cfg["n"]), int(cfg["p"]), int(cfg["k"]), float(cfg["q"])
    if n <= 0 or p <= 0:
        raise ValueError("n and p must be positive integers.")
    if k <= 0 or k > p:
        raise ValueError(f"k must be in [1, p]; got k={k}, p={p}.")
    if int(cfg["n_trials"]) <= 0:
        raise ValueError("n_trials must be positive.")
    if not (0 < q < 1):
        raise ValueError("q must be in (0,1).")
    # Warn (don’t fail) about equi-knockoffs feasibility; enforce later at use site.
    if n < 2 * p:
        warnings.warn(
            f"n ({n}) is less than 2p ({2*p}); equi-knockoff construction may fail. "
            "Increase n or decrease p.",
            RuntimeWarning,
        )

# ------------------------------ Core helpers ----------------------------------

def _trial_seeds(base_seed: int, trial_id: int) -> Tuple[int, int]:
    """
    Deterministic mapping so (base_seed, trial_id) -> (seedA, seedB).
    """
    rng = np.random.default_rng(hash((base_seed, trial_id)) & 0x7FFFFFFF)
    return int(rng.integers(2**31 - 1)), int(rng.integers(2**31 - 1))


# ======================= NEW: PRECOMPUTATION UTILITIES =======================

def compute_theoretical_sigma(p: int, mode: str, rho: float | None = None) -> np.ndarray:
    """
    Compute theoretical covariance matrix for the design.

    For AR(1): Sigma[i,j] = rho^|i-j|
    For IID: Sigma = I

    This can be precomputed once and reused across all trials!
    """
    if mode == "iid":
        return np.eye(p, dtype=np.float64)
    elif mode == "ar1":
        if rho is None:
            raise ValueError("rho required for AR(1)")
        idx = np.arange(p)
        return (rho ** np.abs(idx[:, None] - idx[None, :])).astype(np.float64)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def precompute_scenario_matrices(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Precompute matrices that are constant across all trials in a scenario.

    Returns dict with:
    - 'Sigma_true': Theoretical covariance (if use_true_sigma=True)
    - 'use_sigma': Whether to use it in knockoffs
    """
    precomputed = {}

    if cfg.get("use_true_sigma", False) and cfg.get("precompute_sigma", True):
        p = int(cfg["p"])
        mode = cfg["mode"]
        rho = cfg.get("rho", None)

        try:
            Sigma_true = compute_theoretical_sigma(p, mode, rho)
            precomputed["Sigma_true"] = Sigma_true
            precomputed["use_sigma"] = True

            # Also precompute eigendecomposition for diagnostics
            evals, evecs = np.linalg.eigh(Sigma_true)
            precomputed["Sigma_evals"] = evals
            precomputed["Sigma_rank"] = int(np.sum(evals > 1e-12))

        except Exception as e:
            warnings.warn(f"Could not precompute Sigma: {e}")
            precomputed["use_sigma"] = False
    else:
        precomputed["use_sigma"] = False

    return precomputed

# ----------------------------------- CLI --------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulation study runner (step 3: analyze + figures)")
    p.add_argument("--check-config", action="store_true", help="Load and validate config(s) only.")
    p.add_argument("--simulate", action="store_true", help="Run simulations for the given config(s) and write results/raw/*.csv")
    p.add_argument("--analyze", action="store_true", help="Aggregate results/raw/*_trials.csv into results/summary.csv")
    p.add_argument("--figures", action="store_true", help="Create summary figures from results/summary.csv")
    p.add_argument("--config", "-c", action="append", default=[], help="Path to a JSON config. Repeatable.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1=all cores, 1=sequential)")
    p.add_argument("--no-parallel", action="store_true", help="Force sequential execution")
    p.add_argument("--no-precompute", action="store_true", help="Disable Sigma precomputation")
    return p.parse_args(argv)

# ================================== Imports ===================================

try:
    from src.dgps import generate_full
    from src.methods import (
        knockoffs_equicorr,
        lasso_path_stats,
        knockoff_select,
        bh_select_marginal,
        bh_select_whitened
    )
    from src.metrics import fdp_power
except Exception:
    # In tests we'll monkeypatch these names in this module (src.simulation.*)
    generate_full = None
    knockoffs_equicorr = None
    lasso_path_stats = None
    knockoff_select = None
    bh_select_marginal = None
    fdp_power= None



# ------------------------------ Core trial runner -----------------------------

# def run_one_trial(cfg: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
#     """
#     Run a single trial for both methods (Knockoff+ and BH-marginal).
#     Returns a flat dict suitable for a DataFrame row.
#     """
#     # 1) Data
#     ds_seed = int(cfg["seed"]) + int(trial_id)
#     y, X, beta, meta = generate_full(
#         int(cfg["n"]), int(cfg["p"]),
#         mode=cfg["mode"], rho=cfg.get("rho", None), df=cfg.get("df", math.inf),
#         k=int(cfg["k"]), A=float(cfg["A"]),
#         normalize=bool(cfg.get("normalize", True)), norm_target=cfg.get("norm_target", "sqrt_n"),
#         seed=ds_seed,
#     )
#     true_support = meta["support_indices"]
#
#     # 2) Knockoffs
#     seed_kn, seed_lasso = _trial_seeds(int(cfg["seed"]), int(trial_id))
#     Xk, info_k = knockoffs_equicorr(X, seed=seed_kn)
#
#     # 3) W-stats via lasso path
#     stats = lasso_path_stats(
#         X, y, Xk,
#         n_alphas=int(cfg["n_alphas"]),
#         eps=float(cfg.get("eps", 1e-3)),
#         coef_tol=float(cfg.get("coef_tol", 1e-9)),
#         max_iter=int(cfg.get("max_iter", 10_000)),
#     )
#     W = stats["W"]
#     sel_kn, info_sel = knockoff_select(W, q=float(cfg["q"]), offset=1)  # Knockoff+
#
#     # 4) BH baseline (marginal)
#     sel_bh, info_bh = bh_select_marginal(X, y, q=float(cfg["q"]))
#
#     # BH + BY/log-factor correction
#     sel_by, info_by = bh_select_marginal(X, y, q=float(cfg["q"]), by_correction=True)
#     m_by = fdp_power(true_support, sel_by)
#
#     # BH + whitened z
#     sel_bw, info_bw = bh_select_whitened(X, y, q=float(cfg["q"]))
#     m_bw = fdp_power(true_support, sel_bw)
#
#     # 5) Per-trial metrics
#     m_kn = fdp_power(true_support, sel_kn)
#     m_bh = fdp_power(true_support, sel_bh)
#
#     # 6) Flat row
#     out = dict(
#         # identifiers
#         name=cfg["name"],
#         trial=int(trial_id),
#
#         # DGP snapshot
#         n=int(cfg["n"]), p=int(cfg["p"]), mode=cfg["mode"],
#         rho=(None if cfg["mode"] == "iid" else cfg.get("rho", None)),
#         k_true=int(cfg["k"]), A=float(cfg["A"]),
#
#         # Knockoff+ info
#         method="Knockoff+",
#         #R_kn=m_kn["R"], TP_kn=m_kn["TP"], V_kn=m_kn["V"],
#         R_kn=m_kn["R"], TP_kn=m_kn["T"], V_kn=m_kn["V"],
#         FDP_kn=m_kn["FDP"], Power_kn=m_kn["Power"],
#         T_kn=info_sel.get("T", None), FDPhat_kn=info_sel.get("fdp_hat", None),
#
#         # BH info
#         R_bh=m_bh["R"], TP_bh=m_bh["T"], V_bh=m_bh["V"],
#         FDP_bh=m_bh["FDP"], Power_bh=m_bh["Power"],
#
#         # BY (BH with log-factor)
#         R_by=m_by["R"], TP_by=m_by["T"], V_by=m_by["V"],
#         FDP_by=m_by["FDP"], Power_by=m_by["Power"],
#
#         # BW (BH with whitened z)
#         R_bw=m_bw["R"], TP_bw=m_bw["T"], V_bw=m_bw["V"],
#         FDP_bw=m_bw["FDP"], Power_bw=m_bw["Power"],
#
#         # extras
#         n_pos=len(sel_kn),
#         w_nonzero=int(np.sum(W != 0.0)),
#         seed_dataset=ds_seed,
#         seed_knockoff=seed_kn,
#     )
#     return out

def run_one_trial(cfg: Dict[str, Any], trial_id: int,precomputed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run a single trial with optional precomputed matrices.

    OPTIMIZATION: If precomputed['Sigma_true'] is provided, use it in knockoffs
    construction instead of computing X^T X / n eigendecomposition.
    """
    # 1) Data generation
    ds_seed = int(cfg["seed"]) + int(trial_id)

    df = cfg.get("df", math.inf)
    if df is None or not (df > 0 or math.isinf(df)):
        df = math.inf

    print("DEBUG in worker:", trial_id, "df in cfg:", cfg.get("df"), "df used:", df, file=sys.stderr)

    y, X, beta, meta = generate_full(
        int(cfg["n"]), int(cfg["p"]),
        mode=cfg["mode"], rho=cfg.get("rho", None), df=df,
        k=int(cfg["k"]), A=float(cfg["A"]),
        normalize=bool(cfg.get("normalize", True)),
        norm_target=cfg.get("norm_target", "sqrt_n"),
        seed=ds_seed,
    )

    true_support = meta["support_indices"]

    # 2) Knockoffs with optional precomputed Sigma
    seed_kn, seed_lasso = _trial_seeds(int(cfg["seed"]), int(trial_id))

    # KEY OPTIMIZATION:
    # Use precomputed theoretical Sigma so that knockoffs_equicorr()
    # skips X^T X / n covariance estimation inside each replicate.
    use_true_sigma = None
    if precomputed and precomputed.get("use_sigma", False):
        use_true_sigma = precomputed["Sigma_true"]

    Xk, info_k = knockoffs_equicorr(X, use_true_Sigma=use_true_sigma, seed=seed_kn)
    # SAFETY CHECK: if theoretical Sigma causes non-finite knockoffs, fall back
    if not np.all(np.isfinite(Xk)):
        print("NAN DETECTED in Xk before fallback (trial:", trial_id, ")", file=sys.stderr)
        # if using theoretical sigma, try fallback
        if use_true_sigma is not None:
            print(" -> Falling back to empirical Sigma", file=sys.stderr)
            use_true_sigma = None
            Xk, info_k = knockoffs_equicorr(X, use_true_Sigma=None, seed=seed_kn)

    # Check again
    if not np.all(np.isfinite(Xk)):
        print("NAN persists in Xk after fallback -- investigation needed (trial:", trial_id, ")", file=sys.stderr)
        # Print summary
        print("min/max Xk:", np.nanmin(Xk), np.nanmax(Xk), file=sys.stderr)
        print("Any NaNs:", np.isnan(Xk).sum(), file=sys.stderr)
        print("Any inf:", np.isinf(Xk).sum(), file=sys.stderr)
        # Return a special failure record (avoids killing simulation)
        return dict(
            name=cfg["name"],
            trial=int(trial_id),
            error="Xk_NAN",
        )
    # 3) W-stats via lasso path
    stats = lasso_path_stats(
        X, y, Xk,
        n_alphas=int(cfg.get("n_alphas", 150)),
        eps=float(cfg.get("eps", 1e-3)),
        coef_tol=float(cfg.get("coef_tol", 1e-9)),
        max_iter=int(cfg.get("max_iter", 10_000)),
    )
    W = stats["W"]
    sel_kn, info_sel = knockoff_select(W, q=float(cfg["q"]), offset=1)

    # 4) BH methods
    sel_bh, info_bh = bh_select_marginal(X, y, q=float(cfg["q"]))
    sel_by, info_by = bh_select_marginal(X, y, q=float(cfg["q"]), by_correction=True)
    sel_bw, info_bw = bh_select_whitened(X, y, q=float(cfg["q"]))

    # 5) Metrics
    m_kn = fdp_power(true_support, sel_kn)
    m_bh = fdp_power(true_support, sel_bh)
    m_by = fdp_power(true_support, sel_by)
    m_bw = fdp_power(true_support, sel_bw)

    # 6) Return results
    return dict(
        name=cfg["name"],
        trial=int(trial_id),
        n=int(cfg["n"]), p=int(cfg["p"]), mode=cfg["mode"],
        rho=(None if cfg["mode"] == "iid" else cfg.get("rho", None)),
        k_true=int(cfg["k"]), A=float(cfg["A"]),
        method="Knockoff+",
        R_kn=m_kn["R"], TP_kn=m_kn["T"], V_kn=m_kn["V"],
        FDP_kn=m_kn["FDP"], Power_kn=m_kn["Power"],
        T_kn=info_sel.get("T", None), FDPhat_kn=info_sel.get("fdp_hat", None),
        R_bh=m_bh["R"], TP_bh=m_bh["T"], V_bh=m_bh["V"],
        FDP_bh=m_bh["FDP"], Power_bh=m_bh["Power"],
        R_by=m_by["R"], TP_by=m_by["T"], V_by=m_by["V"],
        FDP_by=m_by["FDP"], Power_by=m_by["Power"],
        R_bw=m_bw["R"], TP_bw=m_bw["T"], V_bw=m_bw["V"],
        FDP_bw=m_bw["FDP"], Power_bw=m_bw["Power"],
        n_pos=len(sel_kn),
        w_nonzero=int(np.sum(W != 0.0)),
        seed_dataset=ds_seed,
        seed_knockoff=seed_kn,
        used_true_sigma=use_true_sigma is not None,
    )

# def run_simulation(cfg: Dict[str, Any]) -> pd.DataFrame:
#     """
#     Run cfg['n_trials'] trials; return a DataFrame with per-trial rows.
#     """
#     try:
#         from tqdm import trange
#     except Exception:
#         trange = range
#
#     rows = []
#     for t in trange(int(cfg["n_trials"]), desc=f"Sim {cfg['name']}"):
#         rows.append(run_one_trial(cfg, t))
#     return pd.DataFrame(rows)


def run_simulation_parallel(
    cfg: Dict[str, Any],
    verbose: bool = True,
    precomputed: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Parallelized simulation with optional precomputed matrices.

    OPTIMIZATION: Precomputed Sigma is shared across all workers (read-only).
    """
    n_trials = int(cfg["n_trials"])
    n_jobs = cfg.get("n_jobs", -1)
    backend = cfg.get("backend", "loky")

    if not HAS_JOBLIB or n_jobs == 1:
        return run_simulation_sequential(cfg, verbose=verbose, precomputed=precomputed)

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Create partial function with precomputed matrices
    trial_func = partial(run_one_trial, cfg, precomputed=precomputed)

    if verbose:
        try:
            from tqdm import tqdm
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(trial_func)(t) for t in tqdm(range(n_trials), desc=f"Sim {cfg['name']}")
            )
        except ImportError:
            print(f"Running {n_trials} trials with {n_jobs} workers...")
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(trial_func)(t) for t in range(n_trials)
            )
    else:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(trial_func)(t) for t in range(n_trials)
        )

    return pd.DataFrame(results)


def run_simulation_sequential(cfg: Dict[str, Any], verbose: bool = True,precomputed: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """Sequential simulation with optional precomputed matrices."""
    try:
        from tqdm import trange
        iterator = trange(int(cfg["n_trials"]), desc=f"Sim {cfg['name']}") if verbose else range(int(cfg["n_trials"]))
    except ImportError:
        iterator = range(int(cfg["n_trials"]))
        if verbose:
            print(f"Running {cfg['n_trials']} trials sequentially...")

    rows = [run_one_trial(cfg, t, precomputed=precomputed) for t in iterator]
    return pd.DataFrame(rows)


def run_simulation(cfg: Dict[str, Any], parallel: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Main simulation entry point with automatic precomputation.

    OPTIMIZATION: Automatically precomputes Sigma if beneficial.
    """
    # Precompute scenario-level matrices
    precomputed = None
    if cfg.get("precompute_sigma", True):
        precomputed = precompute_scenario_matrices(cfg)
        if verbose and precomputed.get("use_sigma", False):
            print(f"  [optimization] Using precomputed theoretical Sigma (rank={precomputed.get('Sigma_rank', '?')})")

    if parallel and HAS_JOBLIB and cfg.get("n_jobs", -1) != 1:
        return run_simulation_parallel(cfg, verbose=verbose, precomputed=precomputed)
    else:
        return run_simulation_sequential(cfg, verbose=verbose, precomputed=precomputed)


# ------------------------------- Aggregation/IO -------------------------------

def save_raw(df: pd.DataFrame, cfg: Dict[str, Any]) -> Path:
    path = RAW_DIR / f"{cfg['name']}_trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    # Save the config alongside
    write_json(cfg, RAW_DIR / f"{cfg['name']}_config.json")
    return path

def _mean_se(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    mask = ~np.isnan(x)
    m = float(np.nanmean(x)) if mask.any() else float("nan")
    se = float(np.nanstd(x[mask], ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else float("nan")
    return m, se


def aggregate_all_raw() -> pd.DataFrame:
    """
    Read all results/raw/*_trials.csv and aggregate per (name, method) into means+SEs.
    Writes results/summary.csv and returns the DataFrame.
    """
    files = sorted(RAW_DIR.glob("*_trials.csv"))
    if not files:
        raise FileNotFoundError(f"No raw trial CSVs found in {RAW_DIR}.")

    summaries: list[dict] = []
    for f in files:
        df = pd.read_csv(f)
        # Expect columns from Step 2 runner
        name = df["name"].iloc[0]
        n = int(df["n"].iloc[0])
        p = int(df["p"].iloc[0])
        mode = df["mode"].iloc[0]
        rho = df["rho"].iloc[0]
        k_true = int(df["k_true"].iloc[0])
        A = float(df["A"].iloc[0])

        # Knockoff+
        fdr_kn, se_fdr_kn = _mean_se(df["FDP_kn"].to_numpy())
        pow_kn, se_pow_kn = _mean_se(df["Power_kn"].to_numpy())
        summaries.append(dict(
            name=name, method="Knockoff+",
            FDR=fdr_kn, FDR_se=se_fdr_kn, Power=pow_kn, Power_se=se_pow_kn,
            n_trials=len(df), n=n, p=p, mode=mode, rho=rho, k_true=k_true, A=A,
        ))

        # BH (marginal)
        fdr_bh, se_fdr_bh = _mean_se(df["FDP_bh"].to_numpy())
        pow_bh, se_pow_bh = _mean_se(df["Power_bh"].to_numpy())
        summaries.append(dict(
            name=name, method="BH (marginal)",
            FDR=fdr_bh, FDR_se=se_fdr_bh, Power=pow_bh, Power_se=se_pow_bh,
            n_trials=len(df), n=n, p=p, mode=mode, rho=rho, k_true=k_true, A=A,
        ))

    out = pd.DataFrame(summaries).sort_values(["name", "method"], ignore_index=True)
    out_path = RESULTS_DIR / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


# ---------------------------------- Figures -----------------------------------

def plot_summary_bars(summary_df: pd.DataFrame, name: str) -> Path:
    """
    Make a side-by-side bar chart (FDR and Power with 95% CIs) for a single config name.
    Saves to results/figures/{name}_fdr_power.png and returns the path.
    """
    sub = summary_df[summary_df["name"] == name].copy()
    if sub.empty:
        raise ValueError(f"No summary rows for name={name!r}.")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)

    for ax, metric, se_col, title in [
        (axes[0], "FDR",   "FDR_se",   "Empirical FDR"),
        (axes[1], "Power", "Power_se", "Empirical Power"),
    ]:
        means = sub[metric].to_numpy()
        ses = sub[se_col].to_numpy()
        labels = sub["method"].tolist()
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=1.96 * ses, capsize=4)
        ax.set_xticks(x, labels, rotation=15)
        ymax = np.nanmax(means + 1.96 * ses) if len(means) else 1.0
        ax.set_ylim(0, max(0.001, float(ymax) * 1.15))
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(title)

    fig.suptitle(f"Simulation summary – {name}", fontsize=12)
    out_path = FIG_DIR / f"{name}_fdr_power.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def make_all_figures(summary_path: Path | None = None) -> list[Path]:
    """
    Read summary.csv and create a figure per unique name.
    """
    if summary_path is None:
        summary_path = RESULTS_DIR / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("summary.csv not found; run --analyze first.")
    df = pd.read_csv(summary_path)
    out_paths: list[Path] = []
    for name in sorted(df["name"].unique()):
        out_paths.append(plot_summary_bars(df, name))
    return out_paths



def main(argv: List[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.check_config:
        if not args.config:
            raise SystemExit("Use -c path/to/config.json (repeatable) with --check-config.")
        for cpath in args.config:
            cfg = load_config(Path(cpath))
            validate_config(cfg)
            print(f"[ok] {cpath}: name={cfg['name']}, n={cfg['n']}, p={cfg['p']}, mode={cfg['mode']}, rho={cfg['rho']}")
        return

    if args.simulate:
        if not args.config:
            raise SystemExit("Use -c path/to/config.json (repeatable) with --simulate.")

        parallel = not args.no_parallel

        for cpath in args.config:
            cfg = load_config(Path(cpath))
            if args.n_jobs != -1:
                cfg["n_jobs"] = args.n_jobs
            if args.no_precompute:
                cfg["precompute_sigma"] = False
            validate_config(cfg)

            print(f"[simulate] Running config: {cfg['name']}")
            print(f"  n={cfg['n']}, p={cfg['p']}, mode={cfg['mode']}, trials={cfg['n_trials']}")

            df = run_simulation(cfg, parallel=parallel)
            out = save_raw(df, cfg)
            print(f"[simulate] Wrote {out}")
        # fallthrough allowed; you can chain --simulate --analyze --figures in one call

    if args.analyze:
        print("[analyze] Aggregating raw trial CSVs...")
        summary = aggregate_all_raw()
        out_path = RESULTS_DIR / "summary.csv"
        print(f"[analyze] Wrote {out_path} with {len(summary)} rows")

    if args.figures:
        print("[figures] Making summary figures...")
        paths = make_all_figures()
        for p in paths:
            print(f"[figures] Wrote {p}")

    if not (args.check_config or args.simulate or args.analyze or args.figures):
        print("Nothing to do. Try --simulate -c configs/baseline.json, then --analyze, then --figures.")


if __name__ == "__main__":
    main()
