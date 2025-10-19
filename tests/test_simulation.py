import pytest
from src.simulation import DEFAULTS, load_config, validate_config

def test_load_config_fills_defaults(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text('{"name":"ok","n":200,"p":50}')
    cfg = load_config(cfg_path)
    # required carried through
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

