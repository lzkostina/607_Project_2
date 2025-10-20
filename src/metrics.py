import numpy as np
from typing import Iterable, Sequence, Union

ArrayLike = Union[Sequence[int], np.ndarray]

def _to_index_set(x: ArrayLike) -> set[int]:
    """
    Convert a variety of inputs to a set of integer indices.
    Supports:
      - iterable of ints (list/tuple/set/ndarray)
      - boolean ndarray mask -> indices via np.flatnonzero
    """
    if isinstance(x, np.ndarray):
        if x.dtype == bool:
            return set(np.flatnonzero(x).astype(int).tolist())
        elif np.issubdtype(x.dtype, np.integer):
            return set(x.astype(int).tolist())
        else:
            # Could be float 0/1 — be strict to avoid silent bugs
            raise TypeError("Array input must be bool mask or integer dtype.")
    # Generic iterables (lists/tuples/sets)
    try:
        ints = [int(i) for i in x]
    except Exception as e:
        raise TypeError("Input must be an iterable of ints or a boolean ndarray.") from e
    return set(ints)

def fdp_power(true_support: ArrayLike, selected: ArrayLike) -> dict:
    """
    Compute per-trial FDP and Power.

    - FDP = V / max(R, 1)
    - Power = TP / |S*|, and is np.nan when |S*| = 0
    """
    S = _to_index_set(true_support)
    R = _to_index_set(selected)

    # Basic sanity checks (optional but helpful)
    if any(i < 0 for i in S | R):
        raise ValueError("Indices must be non-negative.")
    # (You can add an optional 'p' to check an upper bound if desired.)

    TP = len(S & R)
    Rsize = len(R)
    V = Rsize - TP

    FDP = (V / Rsize) if Rsize > 0 else 0.0
    Power = (TP / len(S)) if len(S) > 0 else np.nan

    return {"R": Rsize, "TP": TP, "V": V, "FDP": FDP, "Power": Power}


def fdr_power_all(true_supports: Iterable[ArrayLike], selections: Iterable[ArrayLike]) -> dict:
    """
    Aggregate empirical FDR and Power across trials (mean of per-trial quantities).
    Power is averaged with np.nanmean over trials where |S*|>0.
    """
    FDP_list, POW_list, per_trial = [], [], []

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
