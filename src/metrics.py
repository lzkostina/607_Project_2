import numpy as np
from typing import Iterable, Sequence

def fdp_power(true_support: Sequence[int], selected: Sequence[int]) -> dict:
    """
    Compute per-trial FDP and Power.

    Definitions (per trial)
    -----------------------
    - R  = # selections = |selected|
    - S* = true support (indices of nonzero effects)
    - V  = # false discoveries = |selected \ S*|
    - TP = # true positives    = |selected ∩ S*|
    - FDP = V / max(R, 1)               (define as 0 when R=0)
    - Power = TP / max(|S*|, 1)         (NaN if |S*|=0, see below)

    Edge cases
    ----------
    - If |S*| = 0 (no signals), Power is undefined. We return np.nan.
      Aggregators should use np.nanmean over trials.

    Parameters
    ----------
    true_support : sequence of ints
        Indices of true nonzero coefficients.
    selected : sequence of ints
        Indices selected by a method.

    Returns
    -------
    dict with keys:
        'R', 'V', 'TP', 'FDP', 'Power'
    """
    S = set(map(int, true_support))
    A = set(map(int, selected))

    R = len(A)
    TP = len(A & S)
    V = R - TP

    FDP = V / max(R, 1)
    Power = (TP / len(S)) if len(S) > 0 else np.nan

    return dict(R=R, V=V, TP=TP, FDP=float(FDP), Power=(float(Power) if np.isfinite(Power) else np.nan))


