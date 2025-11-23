import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("results/profiling/complexity_baseline.csv")
OUT_FIG = Path("results/figures/complexity_optimized.png")

def main():
    # Load data
    df = pd.read_csv(CSV_PATH)

    # Extract n and runtime
    n = df["n"].values
    t = df["runtime_sec"].values

    # Safety check
    if len(n) < 2:
        raise ValueError("Need at least two points to plot complexity curve.")

    # Log-transform for convenience
    log_n = np.log(n)
    log_t = np.log(t)

    # Fit slope α in log–log space: log(t) = α log(n) + b
    alpha, intercept = np.polyfit(log_n, log_t, 1)

    # Print complexity estimate
    print(f"Estimated complexity exponent α ≈ {alpha:.3f}")
    print(f"(i.e., runtime ≈ O(n^{alpha:.2f}))")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n, t, "o-", label="Empirical runtime")

    # Plot fitted power-law curve for visualization
    n_fit = np.linspace(n.min(), n.max(), 200)
    t_fit = np.exp(intercept) * n_fit**alpha
    ax.plot(n_fit, t_fit, "--", label=f"Fit: O(n^{alpha:.2f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size n (log scale)")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_title("Empirical Computational Complexity (baseline implementation)")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # Save figure
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    print(f"Saved plot to {OUT_FIG}")

if __name__ == "__main__":
    main()
