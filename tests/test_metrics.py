import numpy as np
import pytest
import math

from src.metrics import fdp_power, fdr_power_all

############################## fdr_power tests ############################

def test_empty_selection_and_no_signals():
    out = fdp_power(true_support=[], selected=[])
    assert out["FDP"] == 0.0
    assert out["Power"] == 0.0

def test_perfect_recovery():
    S = [1,2,3,10]
    out = fdp_power(selected=[1,2,3,10], true_support=S)
    assert out["FDP"] == 0.0
    assert abs(out["Power"] - 1.0) < 1e-12

def test_partial_and_aggregate():
    # trial1: select half of true signals, no false
    S1 = [1,2,3,4]; sel1 = [1,3]
    # trial2: select 1 true and 1 false
    S2 = [5,6]; sel2 = [6, 100]
    agg = fdr_power_all([sel1, sel2], [S1, S2])
    # trial1: FDP=0/2=0, Power=2/4=0.5
    # trial2: FDP=1/2=0.5, Power=1/2=0.5
    assert abs(agg["FDR"] - (0 + 0.5)/2) < 1e-12
    assert abs(agg["Power"] - 0.5) < 1e-12