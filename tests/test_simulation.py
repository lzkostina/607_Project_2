import pytest
from src.simulation import DEFAULTS, load_config, validate_config, _trial_seeds

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

