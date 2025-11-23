from __future__ import annotations
import time
from pathlib import Path
import json
import pandas as pd

from src.simulation import load_config, validate_config, run_simulation

BASE_CONFIG = Path("configs/ar1_rho0p5_p1000_n3000_k10_A3p5.json")

def main() -> None:
    base_cfg = load_config(BASE_CONFIG)
    validate_config(base_cfg)

    # We will vary n but keep p, mode, rho, k, A, q, etc. fixed
    n_values = [2500, 3000, 3500, 4000, 4500, 5000]
    n_trials_small = 50  # smaller than 600 for speed

    rows = []
    for n in n_values:
        cfg = dict(base_cfg)
        cfg["n"] = n
        cfg["name"] = f"complexity_n{n}"
        cfg["n_trials"] = n_trials_small

        validate_config(cfg)
        print(f"[complexity] Running n={n} with n_trials={n_trials_small}...")

        t0 = time.perf_counter()
        _ = run_simulation(cfg)   # no need to save CSVs here
        t1 = time.perf_counter()
        runtime = t1 - t0

        rows.append({"n": n, "n_trials": n_trials_small, "runtime_sec": runtime})
        print(f"runtime = {runtime:.2f} seconds")

    out_path = Path("results/profiling/complexity_baseline.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[complexity] Wrote {out_path}")

if __name__ == "__main__":
    main()
