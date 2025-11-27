from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    df = pd.read_csv("results/profiling/complexity_comparison.csv")

    n = df["n"].to_numpy()
    t_base = df["baseline_sec"].to_numpy()
    t_opt = df["optimized_sec"].to_numpy()

    # log–log plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(n, t_base, "o-", label="Baseline")
    ax.loglog(n, t_opt, "o-", label="Optimized")

    ax.set_xlabel("n (log scale)")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title("Runtime vs n: baseline vs optimized")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    out_path = OUT_DIR / "complexity_baseline_vs_optimized.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
