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


def fdr_power_all(true_supports: Iterable[Sequence[int]], selections: Iterable[Sequence[int]]) -> dict:
    """
    Aggregate empirical FDR and Power across trials.

    We average the *per-trial* quantities:
        FDR_hat  = mean(FDP_t)
        Power_hat= mean(Power_t)   (ignoring trials with no signals via nanmean)

    Parameters
    ----------
    true_supports : iterable of sequences of ints
        One sequence per trial listing true signal indices.
    selections : iterable of sequences of ints
        One sequence per trial listing selected indices.

    Returns
    -------
    dict with keys:
        'FDR'         : mean FDP across trials
        'Power'       : mean Power across trials (nanmean over trials with |S*|>0)
        'FDR_se'      : std(FDP)/sqrt(T)
        'Power_se'    : std(Power)/sqrt(T_pos)   (T_pos = # trials with |S*|>0)
        'per_trial'   : list of dicts from fdp_power_single (for diagnostics)
        'num_trials'  : total number of trials
        'num_trials_pos': number of trials with |S*|>0
    """
    FDP_list = []
    POW_list = []
    per_trial = []

    for S, A in zip(true_supports, selections):
        stats = fdp_power(S, A)
        per_trial.append(stats)
        FDP_list.append(stats["FDP"])
        POW_list.append(stats["Power"])  # may include np.nan

    FDP_arr = np.asarray(FDP_list, dtype=float)
    POW_arr = np.asarray(POW_list, dtype=float)

    T = len(per_trial)
    T_pos = int(np.sum(~np.isnan(POW_arr)))  # trials with |S*|>0

    FDR_hat = float(np.mean(FDP_arr)) if T > 0 else np.nan
    Power_hat = float(np.nanmean(POW_arr)) if T_pos > 0 else np.nan

    FDR_se = float(np.std(FDP_arr, ddof=1) / np.sqrt(T)) if T > 1 else np.nan
    Power_se = float(np.nanstd(POW_arr, ddof=1) / np.sqrt(T_pos)) if T_pos > 1 else np.nan

    return dict(
        FDR=FDR_hat,
        Power=Power_hat,
        FDR_se=FDR_se,
        Power_se=Power_se,
        per_trial=per_trial,
        num_trials=T,
        num_trials_pos=T_pos,
    )