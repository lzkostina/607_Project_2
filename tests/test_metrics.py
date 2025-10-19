import numpy as np
import pytest

from src.metrics import fdp_power

def test_fdp_power_single_basic_and_edges():
    # True signals at {1,3,5}
    S = [1, 3, 5]

    # Case 1: perfect selection
    sel = [1, 3, 5]
    out = fdp_power(S, sel)
    assert out["R"] == 3 and out["TP"] == 3 and out["V"] == 0
    assert out["FDP"] == 0.0 and out["Power"] == 1.0

    # Case 2: some FP, some TP
    sel = [0, 1, 2, 3]
    out = fdp_power(S, sel)
    # TP=2 (1 and 3), V=2 (0 and 2), R=4
    assert out["R"] == 4 and out["TP"] == 2 and out["V"] == 2
    assert abs(out["FDP"] - 0.5) < 1e-12
    assert abs(out["Power"] - (2/3)) < 1e-12

    # Case 3: no selections -> FDP=0 by convention, Power based on TP=0
    sel = []
    out = fdp_power(S, sel)
    assert out["R"] == 0 and out["TP"] == 0 and out["V"] == 0
    assert out["FDP"] == 0.0
    assert out["Power"] == 0.0

    # Case 4: no signals -> Power is NaN
    S0 = []
    sel = [0, 2]
    out = fdp_power(S0, sel)
    assert np.isnan(out["Power"])
    # FDP still well-defined
    assert out["FDP"] == 0.0  # all selections are "false", but R>0 so V/R = 1.0
    # Wait: with S* empty, V=R, so FDP=R/max(R,1)=1.0 if R>0, else 0.0
    # Let's check explicitly:
    R = len(sel)
    expected_fdp = 0.0 if R == 0 else 1.0
    assert abs(out["FDP"] - expected_fdp) < 1e-12

