import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Load .csv files and reshape wide → long
# ---------------------------------------------------------------------

METHOD_MAP = {
    "kn": "Knockoff+",
    "bh": "BHq",
    "by": "BY",
    "bw": "Bonferroni-Holm",
}


def load_and_transform(results_dir: Path) -> pd.DataFrame:
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)

        # Each row corresponds to one trial for ONE method: Knockoff+
        # But we ignore df["method"], because real methods are encoded in suffixes.

        k = df["k_true"]
        trial = df["trial"]

        # Build long-format rows manually
        long_rows = []

        for method_suffix, method_name in METHOD_MAP.items():
            R = df[f"R_{method_suffix}"]
            TP = df[f"TP_{method_suffix}"]
            V = df[f"V_{method_suffix}"]
            FDP = df[f"FDP_{method_suffix}"]
            Power = df[f"Power_{method_suffix}"]

            long_rows.append(pd.DataFrame({
                "k": k,
                "trial": trial,
                "method": method_name,
                "R": R,
                "TP": TP,
                "V": V,
                "FDP": FDP,
                "Power": Power,
            }))

        dfs.append(pd.concat(long_rows, ignore_index=True))

    full = pd.concat(dfs, ignore_index=True)
    return full


# ---------------------------------------------------------------------
# 2. Aggregate results: mean FDP and mean Power per method × k
# ---------------------------------------------------------------------

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["k", "method"], as_index=False)
          .agg(
              FDR=("FDP", "mean"),
              Power=("Power", "mean"),
          )
    )
    return agg


# ---------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------

def plot_fdr_vs_k(agg, q, outpath=None):
    df_piv = agg.pivot(index="k", columns="method", values="FDR").sort_index()

    plt.figure(figsize=(7.5, 6))
    plt.axhline(q, linestyle="--", color="black", label=f"Nominal {100*q:.0f}%")

    for method in df_piv.columns:
        plt.plot(df_piv.index, df_piv[method], marker="o", linewidth=2, label=method)

    plt.ylim(0, 0.35)
    plt.xlabel("Sparsity k")
    plt.ylabel("Mean FDR")
    plt.legend()
    plt.tight_layout()

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300)
        print("Saved:", outpath)
    else:
        plt.show()


def plot_power_vs_k(agg, outpath=None):
    df_piv = agg.pivot(index="k", columns="method", values="Power").sort_index()

    plt.figure(figsize=(7.5, 6))

    for method in df_piv.columns:
        plt.plot(df_piv.index, df_piv[method], marker="o", linewidth=2, label=method)

    plt.ylim(0, 1.05)
    plt.xlabel("Sparsity k")
    plt.ylabel("Mean Power")
    plt.legend()
    plt.tight_layout()

    if outpath:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300)
        print("Saved:", outpath)
    else:
        plt.show()


# ---------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/raw")
    parser.add_argument("--out-dir", type=str, default="results/sec332_figures")
    parser.add_argument("--q", type=float, default=0.20)
    args = parser.parse_args()

    df = load_and_transform(Path(args.results_dir))
    agg = aggregate_results(df)

    out_dir = Path(args.out_dir)

    plot_fdr_vs_k(agg, q=args.q, outpath=out_dir / "fdr_vs_k.png")
    plot_power_vs_k(agg, outpath=out_dir / "power_vs_k.png")


if __name__ == "__main__":
    main()
