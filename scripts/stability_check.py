from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.simulation import load_config, validate_config, run_simulation


CONFIG_DIR = Path("configs")
OUT_DIR = Path("results/profiling")
OUT_CSV = OUT_DIR / "stability_warnings.csv"


def run_one_config(cfg_path: Path, max_trials: int = 100) -> List[Dict]:
    """
    Run a (possibly reduced) simulation for a single config and
    collect all warnings emitted during the run.

    Returns a list of dicts, one per warning.
    """
    cfg = load_config(cfg_path)
    validate_config(cfg)

    # Keep the original n_trials but cap it for the stability check
    orig_trials = int(cfg.get("n_trials", 100))
    n_trials = min(orig_trials, max_trials)
    cfg["n_trials"] = n_trials

    print(f"[stability] Running config={cfg_path.name} with n_trials={n_trials}")

    records: List[Dict] = []

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        # No extra tqdm here: rely on run_simulation's own verbose flag
        _ = run_simulation(cfg, parallel=True, verbose=False)

    # Convert warnings to rows
    for w in wlist:
        records.append(
            dict(
                config=cfg_path.name,
                category=type(w.message).__name__,
                message=str(w.message),
                filename=str(getattr(w, "filename", "")),
                lineno=int(getattr(w, "lineno", -1)),
                orig_n_trials=orig_trials,
                used_n_trials=n_trials,
            )
        )

    if not records:
        print(f"[stability] No warnings for {cfg_path.name}")
    else:
        print(f"[stability] {len(records)} warnings for {cfg_path.name}")

    return records


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_paths = sorted(CONFIG_DIR.glob("*.json"))
    if not cfg_paths:
        print(f"[stability] No configs found in {CONFIG_DIR}/")
        return

    all_rows: List[Dict] = []

    for cfg_path in cfg_paths:
        rows = run_one_config(cfg_path, max_trials=100)
        all_rows.extend(rows)

    if not all_rows:
        print("[stability] No warnings across all configs 🎉")
        # still write an empty CSV for reproducibility
        pd.DataFrame(columns=[
            "config", "category", "message", "filename", "lineno",
            "orig_n_trials", "used_n_trials"
        ]).to_csv(OUT_CSV, index=False)
        print(f"[stability] Wrote empty {OUT_CSV}")
        return

    df = pd.DataFrame(all_rows)

    # Save full warning log
    df.to_csv(OUT_CSV, index=False)
    print(f"[stability] Wrote warning log to {OUT_CSV}")

    # Print a small summary by config + category
    summary = (
        df.groupby(["config", "category"])
          .size()
          .reset_index(name="n_warnings")
          .sort_values(["config", "n_warnings"], ascending=[True, False])
    )
    print("\n[stability] Summary by config + category:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
