import pytest
import numpy as np
import pandas as pd
import types
from pathlib import Path
from src.simulation import *
from src.simulation import _trial_seeds
import src.simulation as sim

def test_load_config_fills_defaults(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"name":"ok","n":200,"p":50}')
    cfg = load_config(cfg_path)
    # to be carried through
    assert cfg["name"] == "ok" and cfg["n"] == 200 and cfg["p"] == 50
    # a known default exists
    assert cfg["q"] == DEFAULTS["q"]
    # mode default
    assert cfg["mode"] == "iid" and cfg["rho"] is None

def test_validate_iid_ok(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"name":"ok","n":200,"p":50,"mode":"iid","rho":null,"k":10,"q":0.2,"n_trials":3,"seed":7}')
    cfg = load_config(cfg_path)
    validate_config(cfg)  # should not raise

def test_validate_ar1_ok(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"name":"ok","n":200,"p":50,"mode":"ar1","rho":0.3,"k":10,"q":0.2,"n_trials":3,"seed":7}')
    cfg = load_config(cfg_path)
    validate_config(cfg)


@pytest.mark.parametrize("bad_mode", ["foo", "", null := None])
def test_bad_mode_raises(tmp_path, bad_mode):
    cfg_path = tmp_path / "cfg.json"
    rho = "null" if bad_mode != "ar1" else "0.5"
    cfg_path.write_text(f'{{"name":"x","n":100,"p":20,"mode":{jsonify(bad_mode)},"rho":null,"k":5,"q":0.2,"n_trials":2,"seed":1}}')
    cfg = load_config(cfg_path)
    with pytest.raises(Exception):
        validate_config(cfg)

def jsonify(x):
    import json
    return json.dumps(x)

def test_ar1_requires_rho(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"name":"x","n":100,"p":20,"mode":"ar1","rho":null,"k":5,"q":0.2,"n_trials":2,"seed":1}')
    cfg = load_config(p)
    with pytest.raises(ValueError):
        validate_config(cfg)

@pytest.mark.parametrize("rho", [-1.5, -1.0, 1.0, 2.0])
def test_bad_rho_range(tmp_path, rho):
    p = tmp_path / "c.json"
    p.write_text(f'{{"name":"x","n":100,"p":20,"mode":"ar1","rho":{rho},"k":5,"q":0.2,"n_trials":2,"seed":1}}')
    cfg = load_config(p)
    with pytest.raises(ValueError):
        validate_config(cfg)

def test_bad_k_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"name":"x","n":100,"p":20,"mode":"iid","rho":null,"k":0,"q":0.2,"n_trials":2,"seed":1}')
    cfg = load_config(p)
    with pytest.raises(ValueError):
        validate_config(cfg)

def test_bad_q_raises(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"name":"x","n":100,"p":20,"mode":"iid","rho":null,"k":5,"q":1.0,"n_trials":2,"seed":1}')
    cfg = load_config(p)
    with pytest.raises(ValueError):
        validate_config(cfg)


############################## _trial_seeds tests ####################################
def test_trial_seeds_deterministic():
    a1 = _trial_seeds(123, 0)
    a2 = _trial_seeds(123, 0)
    assert a1 == a2

def test_trial_seeds_vary_by_trial():
    s0 = _trial_seeds(42, 0)
    s1 = _trial_seeds(42, 1)
    assert s0 != s1

def test_trial_seeds_vary_by_base_seed():
    sA = _trial_seeds(1, 7)
    sB = _trial_seeds(2, 7)
    assert sA != sB

def test_trial_seeds_within_range():
    sA, sB = _trial_seeds(999, 13)
    assert 0 <= sA < 2**31 - 1
    assert 0 <= sB < 2**31 - 1

# ------------------------------ Fakes -----------------------------------------

def fake_generate_full(n, p, mode="iid", rho=None, df=np.inf, k=5, A=3.0, normalize=True, norm_target="sqrt_n", seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    supp = rng.choice(p, size=k, replace=False)
    beta[supp] = A
    y = X @ beta + rng.standard_normal(n)
    meta = {"support_indices": set(map(int, supp))}
    return y, X, beta, meta

def fake_build_knockoffs_equicorr(X, seed=0):
    rng = np.random.default_rng(seed)
    # Simple: knockoff features as permuted columns + tiny noise
    perm = rng.permutation(X.shape[1])
    Xk = X[:, perm] + 1e-6 * rng.standard_normal(X.shape)
    return Xk, {"perm": perm.tolist()}

def fake_lasso_path_stats(X, y, Xk, n_alphas=100, eps=1e-3, coef_tol=1e-9, max_iter=10_000):
    # Very cheap "W": signed correlations difference
    w1 = X.T @ y
    w2 = Xk.T @ y
    W = np.abs(w1) - np.abs(w2)
    return {"W": W}

def fake_knockoff_select(W, q=0.2, offset=1):
    # Threshold by empirical quantile to mimic FDR-ish behavior
    thr = np.quantile(np.abs(W), 1 - q)
    sel = set(np.where(np.abs(W) >= thr)[0].tolist())
    return sel, {"T": float(thr), "fdp_hat": float(q)}

def fake_bh_select_marginal(X, y, q=0.2, by_correction=False):
    # Select top-|corr| variables up to ~q proportion
    scores = np.abs(X.T @ y)
    k = max(1, int(q * X.shape[1]))
    idx = np.argsort(scores)[-k:]
    return set(map(int, idx.tolist())), {"k": k}

def fake_fdp_power_single(true_support, selected):
    true_support = set(true_support)
    selected = set(selected)
    R = len(selected)
    TP = len(true_support & selected)
    V = R - TP
    FDP = (V / R) if R > 0 else 0.0
    Power = (TP / len(true_support)) if true_support else 0.0
    return {"R": R, "TP": TP, "V": V, "FDP": FDP, "Power": Power}

# ------------------------------ Tests -----------------------------------------

def test_run_one_trial_monkeypatched(monkeypatch, tmp_path):
    # Patch the heavy symbols in the module under test
    monkeypatch.setattr(sim, "generate_full", fake_generate_full, raising=True)
    monkeypatch.setattr(sim, "knockoffs_equicorr", fake_build_knockoffs_equicorr, raising=True)
    monkeypatch.setattr(sim, "lasso_path_stats", fake_lasso_path_stats, raising=True)
    monkeypatch.setattr(sim, "knockoff_select", fake_knockoff_select, raising=True)
    monkeypatch.setattr(sim, "bh_select_marginal", fake_bh_select_marginal, raising=True)
    monkeypatch.setattr(sim, "fdp_power", fake_fdp_power_single, raising=True)

    cfg = {
        "name": "tmini",
        "n": 60,
        "p": 20,
        "mode": "iid",
        "rho": None,
        "k": 5,
        "A": 3.0,
        "q": 0.2,
        "n_alphas": 50,
        "n_trials": 2,
        "seed": 2025,
        "normalize": True,
        "norm_target": "sqrt_n",
    }
    row = sim.run_one_trial(cfg, trial_id=0)
    # Basic schema checks
    for key in ["name", "trial", "n", "p", "mode", "k_true", "A", "FDP_kn", "Power_kn", "FDP_bh", "Power_bh"]:
        assert key in row
    assert row["name"] == "tmini"
    assert row["trial"] == 0
    assert 0 <= row["FDP_kn"] <= 1
    assert 0 <= row["Power_kn"] <= 1

def test_run_simulation_writes_raw(monkeypatch, tmp_path):
    # Monkeypatch same as above
    monkeypatch.setattr(sim, "generate_full", fake_generate_full, raising=True)
    monkeypatch.setattr(sim, "knockoffs_equicorr", fake_build_knockoffs_equicorr, raising=True)
    monkeypatch.setattr(sim, "lasso_path_stats", fake_lasso_path_stats, raising=True)
    monkeypatch.setattr(sim, "knockoff_select", fake_knockoff_select, raising=True)
    monkeypatch.setattr(sim, "bh_select_marginal", fake_bh_select_marginal, raising=True)
    monkeypatch.setattr(sim, "fdp_power", fake_fdp_power_single, raising=True)

    # Redirect results dir for this test (optional). Here we just run and check DF shape.
    cfg = {
        "name": "tmini",
        "n": 50,
        "p": 15,
        "mode": "iid",
        "rho": None,
        "k": 4,
        "A": 2.5,
        "q": 0.25,
        "n_alphas": 40,
        "n_trials": 3,
        "seed": 7,
        "normalize": True,
        "norm_target": "sqrt_n",
    }
    df = sim.run_simulation(cfg)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert {"FDP_kn", "Power_kn", "FDP_bh", "Power_bh"} <= set(df.columns)

def _write_fake_raw(tmp_raw: Path, name: str, rows: list[dict]) -> Path:
    tmp_raw.mkdir(parents=True, exist_ok=True)
    p = tmp_raw / f"{name}_trials.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_aggregate_all_raw_from_files(tmp_path, monkeypatch):
    # point module dirs to tmp
    monkeypatch.setattr(sim, "RESULTS_DIR", tmp_path / "results", raising=False)
    monkeypatch.setattr(sim, "RAW_DIR", sim.RESULTS_DIR / "raw", raising=False)
    monkeypatch.setattr(sim, "FIG_DIR", sim.RESULTS_DIR / "figures", raising=False)
    sim.RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Create a minimal raw file that matches the Step 2 schema
    rows = [
        dict(name="cfgA", trial=0, n=100, p=20, mode="iid", rho=None, k_true=5, A=3.5,
             FDP_kn=0.10, Power_kn=0.60, FDP_bh=0.20, Power_bh=0.40),
        dict(name="cfgA", trial=1, n=100, p=20, mode="iid", rho=None, k_true=5, A=3.5,
             FDP_kn=0.20, Power_kn=0.50, FDP_bh=0.30, Power_bh=0.30),
    ]
    _write_fake_raw(sim.RAW_DIR, "cfgA", rows)

    df = sim.aggregate_all_raw()
    assert {"name","method","FDR","FDR_se","Power","Power_se","n_trials"} <= set(df.columns)
    # Two methods → two rows for one config
    assert len(df) == 2
    # Means should match hand-calcs
    kn = df[df["method"]=="Knockoff+"].iloc[0]
    bh = df[df["method"]=="BH (marginal)"].iloc[0]
    assert abs(kn["FDR"]   - 0.15) < 1e-9
    assert abs(kn["Power"] - 0.55) < 1e-9
    assert abs(bh["FDR"]   - 0.25) < 1e-9
    assert abs(bh["Power"] - 0.35) < 1e-9
    assert kn["n_trials"] == 2 and bh["n_trials"] == 2


def test_plot_summary_bars_creates_file(tmp_path, monkeypatch):
    # point module dirs to tmp
    monkeypatch.setattr(sim, "FIG_DIR", tmp_path / "figs", raising=False)

    # Minimal summary for one config
    summary = pd.DataFrame({
        "name": ["cfgA","cfgA"],
        "method": ["Knockoff+","BH (marginal)"],
        "FDR": [0.1, 0.25], "FDR_se":[0.02, 0.03],
        "Power":[0.6, 0.35], "Power_se":[0.04, 0.05],
    })
    out = sim.plot_summary_bars(summary, name="cfgA")
    assert out.exists()
    assert out.name == "cfgA_fdr_power.png"

def test_make_all_figures_reads_summary(tmp_path, monkeypatch):
    # point module dirs to tmp and prepare summary.csv
    monkeypatch.setattr(sim, "RESULTS_DIR", tmp_path / "results", raising=False)
    monkeypatch.setattr(sim, "FIG_DIR", sim.RESULTS_DIR / "figures", raising=False)
    sim.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sim.FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "name": ["cfgA","cfgA"],
        "method": ["Knockoff+","BH (marginal)"],
        "FDR": [0.1, 0.25], "FDR_se":[0.02, 0.03],
        "Power":[0.6, 0.35], "Power_se":[0.04, 0.05],
    })
    summary_path = sim.RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)

    outs = sim.make_all_figures()
    assert len(outs) == 1
    assert outs[0].exists()
