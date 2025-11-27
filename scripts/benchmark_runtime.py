from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

from src.simulation import load_config, validate_config, run_simulation

PROFILE_CONFIG = Path("configs/ar1_rho0p5_p1000_n3000_k10_A3p5.json")


def main() -> None:
    cfg_base = load_config(PROFILE_CONFIG)
    validate_config(cfg_base)

    # Use a manageable number of trials for the benchmark
    n_trials = int(cfg_base.get("n_trials", 100))
    cfg_base["n_trials"] = n_trials

    print(f"[benchmark] Config={PROFILE_CONFIG.name}, n_trials={n_trials}")

    # ----------------------------
    # Sequential benchmark
    # ----------------------------
    cfg_seq = dict(cfg_base)
    cfg_seq["n_jobs"] = 1

    print("[benchmark] Running sequential...", flush=True)
    t0 = time.perf_counter()
    _ = run_simulation(cfg_seq, parallel=False, verbose=True)
    t1 = time.perf_counter()
    t_seq = t1 - t0

    print(f"[benchmark] Sequential runtime: {t_seq:.2f} s")

    # ----------------------------
    # Parallel benchmark
    # ----------------------------
    cfg_par = dict(cfg_base)
    cfg_par["n_jobs"] = -1  # use all cores

    print("[benchmark] Running parallel...", flush=True)
    t2 = time.perf_counter()
    _ = run_simulation(cfg_par, parallel=True, verbose=True)
    t3 = time.perf_counter()
    t_par = t3 - t2

    print(f"[benchmark] Parallel runtime:   {t_par:.2f} s")

    # ----------------------------
    # Speedup summary
    # ----------------------------
    speedup = t_seq / t_par if t_par > 0 else float("nan")
    print(f"[benchmark] Speedup (seq/par): {speedup:.2f}x")

    # ----------------------------
    # Save benchmark CSV
    # ----------------------------
    out_dir = Path("results/profiling")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_runtime.csv"

    pd.DataFrame(
        [{
            "config": PROFILE_CONFIG.name,
            "n_trials": n_trials,
            "t_seq_sec": t_seq,
            "t_par_sec": t_par,
            "speedup": speedup,
        }]
    ).to_csv(out_path, index=False)

    print(f"[benchmark] Wrote {out_path}")


if __name__ == "__main__":
    main()
