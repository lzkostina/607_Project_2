from __future__ import annotations

import numpy as np
import pandas as pd
import tqdm
import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple



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
}


def load_config(path: Path) -> Dict[str, Any]:
    """
    Read JSON config and fill in missing fields from DEFAULTS.
    Required keys: name, n, p. Others default.
    """
    user = read_json(path)
    for key in ("name", "n", "p"):
        if key not in user:
            raise KeyError(f"Config {path} missing required key: {key!r}")
    cfg = {**DEFAULTS, **user}  # user overrides defaults
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


# ----------------------------------- CLI --------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulation study runner (step 2: trial loop + raw CSV)")
    p.add_argument("--check-config", action="store_true", help="Load and validate config(s) only.")
    p.add_argument("--simulate", action="store_true", help="Run simulations for the given config(s) and write results/raw/*.csv")
    p.add_argument("--config", "-c", action="append", default=[], help="Path to a JSON config. Repeatable.")
    return p.parse_args(argv)


try:
    from src.dgps import generate_full
    from src.methods import (
        knockoffs_equicorr,
        lasso_path_stats,
        knockoff_select,
        bh_select_marginal,
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

def run_one_trial(cfg: Dict[str, Any], trial_id: int) -> Dict[str, Any]:
    """
    Run a single trial for both methods (Knockoff+ and BH-marginal).
    Returns a flat dict suitable for a DataFrame row.
    """
    # Safety: these must be available at runtime
    required_syms = [
        ("generate_full", generate_full),
        ("knockoffs_equicorr", knockoffs_equicorr),
        ("lasso_path_stats", lasso_path_stats),
        ("knockoff_select", knockoff_select),
        ("bh_select_marginal", bh_select_marginal),
        ("fdp_power", fdp_power),
    ]
    missing = [name for name, sym in required_syms if sym is None]
    if missing:
        raise RuntimeError(f"Missing required symbols (did you install your project code?): {missing}")

    # 1) Data
    ds_seed = int(cfg["seed"]) + int(trial_id)
    y, X, beta, meta = generate_full(
        int(cfg["n"]), int(cfg["p"]),
        mode=cfg["mode"], rho=cfg.get("rho", None), df=cfg.get("df", math.inf),
        k=int(cfg["k"]), A=float(cfg["A"]),
        normalize=bool(cfg.get("normalize", True)), norm_target=cfg.get("norm_target", "sqrt_n"),
        seed=ds_seed,
    )
    true_support = meta["support_indices"]

    # 2) Knockoffs
    seed_kn, seed_lasso = _trial_seeds(int(cfg["seed"]), int(trial_id))
    Xk, info_k = knockoffs_equicorr(X, seed=seed_kn)

    # 3) W-stats via lasso path
    stats = lasso_path_stats(
        X, y, Xk,
        n_alphas=int(cfg["n_alphas"]),
        eps=float(cfg.get("eps", 1e-3)),
        coef_tol=float(cfg.get("coef_tol", 1e-9)),
        max_iter=int(cfg.get("max_iter", 10_000)),
    )
    W = stats["W"]
    sel_kn, info_sel = knockoff_select(W, q=float(cfg["q"]), offset=1)  # Knockoff+

    # 4) BH baseline (marginal)
    sel_bh, info_bh = bh_select_marginal(X, y, q=float(cfg["q"]))

    # 5) Per-trial metrics
    m_kn = fdp_power(true_support, sel_kn)
    m_bh = fdp_power(true_support, sel_bh)

    # 6) Flat row
    out = dict(
        # identifiers
        name=cfg["name"],
        trial=int(trial_id),

        # DGP snapshot
        n=int(cfg["n"]), p=int(cfg["p"]), mode=cfg["mode"],
        rho=(None if cfg["mode"] == "iid" else cfg.get("rho", None)),
        k_true=int(cfg["k"]), A=float(cfg["A"]),

        # Knockoff+ info
        method="Knockoff+",
        R_kn=m_kn["R"], TP_kn=m_kn["TP"], V_kn=m_kn["V"],
        FDP_kn=m_kn["FDP"], Power_kn=m_kn["Power"],
        T_kn=info_sel.get("T", None), FDPhat_kn=info_sel.get("fdp_hat", None),

        # BH info
        R_bh=m_bh["R"], TP_bh=m_bh["TP"], V_bh=m_bh["V"],
        FDP_bh=m_bh["FDP"], Power_bh=m_bh["Power"],

        # extras
        n_pos=len(sel_kn),
        w_nonzero=int(np.sum(W != 0.0)),
        seed_dataset=ds_seed,
        seed_knockoff=seed_kn,
    )
    return out


def run_simulation(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Run cfg['n_trials'] trials; return a DataFrame with per-trial rows.
    """
    try:
        from tqdm import trange
    except Exception:
        trange = range

    rows = []
    for t in trange(int(cfg["n_trials"]), desc=f"Sim {cfg['name']}"):
        rows.append(run_one_trial(cfg, t))
    return pd.DataFrame(rows)


# ------------------------------- Aggregation/IO -------------------------------

def save_raw(df: pd.DataFrame, cfg: Dict[str, Any]) -> Path:
    path = RAW_DIR / f"{cfg['name']}_trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    # Save the config alongside
    write_json(cfg, RAW_DIR / f"{cfg['name']}_config.json")
    return path


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
        for cpath in args.config:
            cfg = load_config(Path(cpath))
            validate_config(cfg)
            print(f"[simulate] Running config: {cfg['name']}")
            df = run_simulation(cfg)
            out = save_raw(df, cfg)
            print(f"[simulate] Wrote {out}")
        return

    print("Nothing to do. Try --simulate -c configs/baseline.json or --check-config -c …")

if __name__ == "__main__":
    main()
