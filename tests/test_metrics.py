import numpy as np
import pytest

from src.metrics import fdp_power, fdr_power_all

############################## fdr_power tests ############################
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
    assert out["R"] == 2 and out["TP"] == 0 and out["V"] == 2
    assert out["FDP"] == 1.0  # since V=R=2


############################## fdr_power_all tests ############################

def test_fdr_power_all_basic():
    # Two trials with signals, one without
    true_list = [
        [1, 3, 5],          # trial 1  (|S*|=3)
        [2, 4],             # trial 2  (|S*|=2)
        [],                 # trial 3  (no signals -> Power NaN)
    ]
    sel_list = [
        [1, 3],             # TP=2,R=2 -> FDP=0, Power=2/3
        [2, 7, 9],          # TP=1,R=3 -> FDP=2/3, Power=1/2
        [0, 1],             # no signals: FDP=1.0, Power=NaN
    ]

    out = fdr_power_all(true_list, sel_list)
    # FDR = mean(FDP) over all three trials
    fdp_vals = [0.0, 2/3, 1.0]
    expected_FDR = sum(fdp_vals) / 3
    assert abs(out["FDR"] - expected_FDR) < 1e-12

    # Power = mean over trials with signals only (nanmean)
    pow_vals = [2/3, 1/2]  # (exclude NaN from the third trial)
    expected_Power = sum(pow_vals) / 2
    assert abs(out["Power"] - expected_Power) < 1e-12

    # SEs defined (nan if insufficient trials)
    assert out["num_trials"] == 3
    assert out["num_trials_pos"] == 2
    assert np.isfinite(out["FDR_se"])
    assert np.isfinite(out["Power_se"])