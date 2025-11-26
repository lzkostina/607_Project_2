from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
from tqdm import tqdm

from src.simulation import load_config, validate_config, run_simulation


# same baseline config
BASE_CONFIG = Path("configs/ar1_rho0p5_p1000_n3000_k10_A3p5.json")


def _log(msg: str) -> None:
    """Print messages only when pytest is run with -s."""
    print(f"[regtest] {msg}")


def _make_small_cfg() -> Dict[str, Any]:
    """
    Build a small regression-test configuration from the baseline JSON,
    with fewer trials so tests run quickly.
    """
    if not BASE_CONFIG.exists():
        pytest.skip(f"Missing baseline config: {BASE_CONFIG}")

    cfg = load_config(BASE_CONFIG)
    validate_config(cfg)

    # Make a small, fast regression config
    cfg = dict(cfg)  # copy
    cfg["name"] = "regtest_ar1_small"
    cfg["n_trials"] = 20        # fewer trials for speed
    cfg["seed"] = 12345         # fixed seed for determinism
    cfg["n_jobs"] = 1           # explicit for the sequential run
    return cfg


def _run_cfg_with_progress(cfg: Dict[str, Any], desc: str) -> pd.DataFrame:
    """
    Run run_simulation(cfg) with a progress bar over trials.
    This wraps run_simulation by temporarily modifying tqdm behavior.
    """
    validate_config(cfg)

    # Monkeypatch tqdm for the duration of this call to give a custom desc.
    _log(f"Running: {desc} (n_trials={cfg['n_trials']}, n_jobs={cfg.get('n_jobs')})")

    # If your run_simulation internally uses tqdm(trange), the bars will show automatically.
    df = run_simulation(cfg)

    _log(f"Finished: {desc}")
    return df


def _sort_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by 'trial' if present, so sequential / parallel orders match.
    """
    if "trial" in df.columns:
        return df.sort_values("trial").reset_index(drop=True)
    return df.reset_index(drop=True)


def _numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


# --------------- Main regression test: sequential vs parallel ---------------

def test_sequential_vs_parallel_equivalence():
    """
    Regression test: the optimized (parallel) implementation should produce
    statistically equivalent results to the sequential baseline implementation.

    We compare summary statistics (mean FDP, mean Power, etc.) to test it.
    """
    cfg_base = _make_small_cfg()

    # --- Sequential (baseline) ---
    cfg_seq = dict(cfg_base)
    cfg_seq["name"] = "regtest_ar1_seq"
    cfg_seq["n_jobs"] = 1
    df_seq = _run_cfg_with_progress(cfg_seq, "Sequential run (baseline)")

    # --- Parallel (optimized) ---
    cfg_par = dict(cfg_base)
    cfg_par["name"] = "regtest_ar1_par"
    cfg_par["n_jobs"] = -1  # all cores
    df_par = _run_cfg_with_progress(cfg_par, "Parallel run (optimized)")

    _log("Parallel R_kn value counts:\n" + repr(df_par["R_kn"].value_counts()))
    _log("Parallel R_bh value counts:\n" + repr(df_par["R_bh"].value_counts()))
    _log("Parallel R_by value counts:\n" + repr(df_par["R_by"].value_counts()))
    _log("Parallel R_bw value counts:\n" + repr(df_par["R_bw"].value_counts()))

    # Basic structural checks
    assert df_seq.shape == df_par.shape
    assert set(df_seq.columns) == set(df_par.columns)

    # Focus on key performance metrics
    metrics = [
        "FDP_kn", "Power_kn",
        "FDP_bh", "Power_bh",
        "FDP_by", "Power_by",
        "FDP_bw", "Power_bw",
    ]
    for col in metrics:
        assert col in df_seq.columns, f"Missing column {col} in sequential results"
        assert col in df_par.columns, f"Missing column {col} in parallel results"

    # Compute means for each metric
    means_seq = df_seq[metrics].mean()
    means_par = df_par[metrics].mean()

    _log("Sequential means:\n" + repr(means_seq))
    _log("Parallel means:\n" + repr(means_par))

    # Check that summary statistics match within a reasonable tolerance
    # (tolerance can be tuned; here 1e-3 is typically plenty tight)
    diff = (means_seq - means_par).abs()
    _log("Absolute differences in means:\n" + repr(diff))

    assert (diff < 1e-3).all(), (
        "Sequential vs parallel summary metrics differ more than tolerance.\n"
        f"Differences:\n{diff}"
    )
    _log("Sequential vs parallel summary metrics match within tolerance.")


# --------------- Second regression: small iid scenario (edge-ish case) ---------------

def test_iid_regression_small():
    """
    Additional regression check on a small iid scenario to ensure that
    the same seeding and parallel infrastructure behave sensibly in a
    different mode.
    """
    if not BASE_CONFIG.exists():
        pytest.skip(f"Missing baseline config: {BASE_CONFIG}")

    cfg = load_config(BASE_CONFIG)
    validate_config(cfg)

    cfg = dict(cfg)
    cfg["name"] = "regtest_iid_small"
    cfg["mode"] = "iid"
    cfg["rho"] = None
    cfg["n"] = 500
    cfg["p"] = 200
    cfg["n_trials"] = 10
    cfg["seed"] = 1532

    # Sequential run
    cfg_seq = dict(cfg)
    cfg_seq["n_jobs"] = 1
    df_seq = _run_cfg_with_progress(cfg_seq, "IID sequential run")

    # Parallel run
    cfg_par = dict(cfg)
    cfg_par["n_jobs"] = -1
    df_par = _run_cfg_with_progress(cfg_par, "IID parallel run")

    assert df_seq.shape == df_par.shape

    df_seq = _sort_trials(df_seq)
    df_par = _sort_trials(df_par)

    num_cols = _numeric_cols(df_seq)

    _log("Comparing iid numeric outputs...")
    pd.testing.assert_frame_equal(
        df_seq[num_cols],
        df_par[num_cols],
        check_dtype=False,
        atol=1e-10,
        rtol=1e-10,
    )
    _log("IID sequential vs parallel results match.")

