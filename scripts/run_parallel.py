from __future__ import annotations
from pathlib import Path

from src.simulation import load_config, validate_config, run_simulation

PROFILE_CONFIG = Path("configs/ar1_rho0p5_p1000_n3000_k10_A3p5.json")

def main() -> None:
    cfg = load_config(PROFILE_CONFIG)
    validate_config(cfg)

    # Use all cores
    cfg["n_jobs"] = -1
    cfg["name"] = cfg.get("name", "parallel_run")

    print(f"[parallel] Running config={PROFILE_CONFIG.name} with n_jobs={cfg['n_jobs']} (parallel=True)")
    df = run_simulation(cfg, parallel=True, verbose=True)

    out_dir = Path("results/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg['name']}_trials_parallel.csv"
    df.to_csv(out_path, index=False)
    print(f"[parallel] Wrote {out_path}")

if __name__ == "__main__":
    main()
