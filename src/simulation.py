from __future__ import annotations

import numpy as np
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
    p = argparse.ArgumentParser(description="Simulation study runner (step 1: config & seeds, no classes)")
    p.add_argument("--check-config", action="store_true", help="Load and validate config(s) only.")
    p.add_argument("--config", "-c", action="append", default=[], help="Path to a JSON config. Repeatable.")
    return p.parse_args(argv)


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

    print("Nothing to do. Try --check-config -c configs/example.json")


if __name__ == "__main__":
    main()
