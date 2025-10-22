import numpy as np
from typing import Iterable, Sequence, Union, Dict

ArrayLike = Union[Sequence[int], np.ndarray]

def fdp_power(true_support: ArrayLike, selected: ArrayLike) -> Dict[str, float]:
    """
    Compute per-trial FDP and Power.
    FDP = V/R with convention FDP=0 when R=0.
    Power = T/|S| with convention Power=0 when |S|=0.
    """

    sel = np.asarray(selected, dtype=int)
    S = np.asarray(true_support, dtype=int)
    R = sel.size
    k = S.size

    if R == 0:
        FDP = 0.0
        POWER = 0.0 if k > 0 else 0.0  # define as 0 when no selections
        return {"FDP": FDP, "Power": POWER, "R": 0, "T": 0}

    # True/False discovery counts
    isin = np.isin(sel, S)
    T = int(isin.sum())
    V = int(R - T)

    FDP = V / R
    POWER = (T / k) if k > 0 else 0.0
    return {"FDP": FDP, "Power": POWER, "R": R, "T": T}


def fdr_power_all(true_supports: Iterable[ArrayLike], selections: Iterable[ArrayLike]) -> dict:
    """
    Aggregate empirical FDR and Power across trials (mean of per-trial quantities).
    Power is averaged with np.nanmean over trials where |S*|>0.
    """
    fdp_vals, pow_vals, Rs, Ts = [], [], [], []
    for sel, sup in zip(selections, true_supports):
        out = fdp_power(sup, sel)
        fdp_vals.append(out["FDP"])
        pow_vals.append(out["Power"])
        Rs.append(out["R"]);
        Ts.append(out["T"])
    return {
        "FDR": float(np.mean(fdp_vals)) if fdp_vals else np.nan,
        "Power": float(np.mean(pow_vals)) if pow_vals else np.nan,
        "R_mean": float(np.mean(Rs)) if Rs else 0.0,
        "T_mean": float(np.mean(Ts)) if Ts else 0.0,
    }
