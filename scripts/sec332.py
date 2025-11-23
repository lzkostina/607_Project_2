import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import t
import matplotlib.pyplot as plt

import knockpy
from knockpy import dgp, knockoffs, knockoff_stats


# ---------- config ----------
seed = 123
rng = np.random.default_rng(seed)
n, p = 3000, 1000
rho = 0.5                   # AR(1) correlation
q = 0.20                    # nominal FDR level
A = 3.5                     # signal amplitude
ks = list(range(10, 201, 10))
nsim = 200                  # increase to 500/1000 for smoother curves

# ---------- helpers ----------
def sample_data(k):
    """Generate (X, y, beta, support, Sigma) with AR(1) Σ, k nonzeros at amplitude A."""
    # AR(1) covariance (p x p), entries: rho**|i-j|
    Sigma = dgp.AR1(p=p, rho=rho)          # <-- no .Sigma
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)

    support = rng.choice(p, size=k, replace=False)
    beta = np.zeros(p)
    beta[support] = A * rng.choice([-1.0, 1.0], size=k)
    y = X @ beta + rng.normal(0, 1, size=n)
    return X, y, beta, support, Sigma


def bhq(pvals, q):
    """Benjamini–Hochberg on vector of p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, m+1) / m)
    passed = np.where(pvals[order] <= thresh)[0]
    if passed.size == 0:
        return np.array([], dtype=int)
    kmax = passed.max()
    sel = order[:kmax+1]
    return np.sort(sel)

def fdp_power(selections, support):
    """Return FDP and Power for one run."""
    R = len(selections)
    if R == 0:
        return 0.0, 0.0
    V = np.setdiff1d(selections, support, assume_unique=False).size
    T = np.intersect1d(selections, support, assume_unique=False).size
    FDP = V / R
    Power = T / len(support)
    return FDP, Power

def marginal_pvals(X, y):
    """Univariate t-tests for each feature j: y = alpha + beta_j X_j + noise."""
    # Center (optional): add intercept analytically
    x = X - X.mean(axis=0, keepdims=True)
    y0 = y - y.mean()
    # slope and residuals per feature (vectorized)
    # beta_j = (x_j^T y) / (x_j^T x_j)
    xy = x.T @ y0                 # (p,)
    xx = np.sum(x**2, axis=0)     # (p,)
    beta = xy / xx
    yhat = x * beta               # contribution per feature
    # residual variance for simple regression: s2 = (||y||^2 - beta*xy)/(n-2)
    y_norm2 = np.sum(y0**2)
    s2 = (y_norm2 - beta * xy) / (n - 2)
    se = np.sqrt(s2 / xx)
    tstat = beta / se
    pvals = 2 * (1 - t.cdf(np.abs(tstat), df=n-2))
    return pvals

def make_knockoffs(X, Sigma):
    # Some versions need both X and Sigma on init
    ksampler = knockoffs.GaussianSampler(X=X, Sigma=Sigma)  # method can default to 'equi' internally

    # Try the two common APIs
    if hasattr(ksampler, "sample_knockoffs"):
        try:
            return ksampler.sample_knockoffs()     # newer API: no args
        except TypeError:
            return ksampler.sample_knockoffs(X)    # some builds still expect X
    elif hasattr(ksampler, "sample"):
        try:
            return ksampler.sample()               # older API: no args
        except TypeError:
            return ksampler.sample(X)              # fallback if it expects X
    else:
        raise RuntimeError("Unknown knockpy sampling API for GaussianSampler.")

def compute_W_lcd(X, Xk, y):
    """
    Lasso coefficient-difference (LCD) knockoff statistic,
    implemented via knockpy.knockoff_stats.LassoStatistic.
    This corresponds to the 'lasso' / 'lcd' choice in KnockoffFilter.
    """
    stat = knockoff_stats.LassoStatistic()
    # zstat="coef" → use lasso coefficients
    # antisym="cd" → |beta_j| - |beta_j^k| (coefficient difference)
    # group_agg irrelevant here since groups=None in non-grouped case
    W = stat.fit(
        X, Xk, y,
        zstat="coef",
        antisym="cd",
        group_agg="sum",   # or "avg" – just rescales, doesn’t change selections
        cv_score=False,
    )
    return W

# ---------- main simulation ----------
out = []
for k in tqdm(ks, desc="k grid"):
    fdp_K, pow_K = [], []
    fdp_Kp, pow_Kp = [], []
    fdp_BH, pow_BH = [], []

    for _ in range(nsim):
        X, y, beta, support, Sigma = sample_data(k)

        Xk = make_knockoffs(X, Sigma)

        # Lasso coefficient difference W stats
        #W = knockoff_stats.LassoStatistic(X, Xk, y, alphas=None, fit_intercept=True, random_state=0)

        W = compute_W_lcd(X, Xk, y)

        # data-dependent threshold for Knockoff (<=) and Knockoff+ (<)
        def knock_select(W, q, plus=False):
            ts = np.sort(np.unique(np.abs(W)))
            sel = []
            for tval in ts:
                num = np.sum(W <= -tval)
                den = np.sum(W >=  tval)
                if plus:
                    crit = (1 + num) / max(den, 1)
                else:
                    crit = num / max(den, 1)
                if crit <= q:
                    sel = np.where(W >= tval)[0]
                    break
            return sel

        sel_K  = knock_select(W, q, plus=False)
        sel_Kp = knock_select(W, q, plus=True)

        # BHq on marginal p-values
        pvals = marginal_pvals(X, y)
        sel_BH = bhq(pvals, q=q)

        # metrics
        for sel, fs, ps in [(sel_K, fdp_K, pow_K),
                            (sel_Kp, fdp_Kp, pow_Kp),
                            (sel_BH, fdp_BH, pow_BH)]:
            FDP, POWER = fdp_power(sel, support)
            fs.append(FDP); ps.append(POWER)

    out.append({
        "k": k,
        "FDR_knock": np.mean(fdp_K),
        "FDR_kplus": np.mean(fdp_Kp),
        "FDR_BHq":   np.mean(fdp_BH),
        "Pow_knock": np.mean(pow_K),
        "Pow_kplus": np.mean(pow_Kp),
        "Pow_BHq":   np.mean(pow_BH),
    })

df = pd.DataFrame(out)
print(df.round(3))

# ---------- plot ----------
plt.figure(figsize=(7.5, 6))
plt.axhline(100*q, linestyle="--", linewidth=1, label="Nominal level")
plt.plot(df["k"], 100*df["FDR_knock"], marker="s", linewidth=2, label="Knockoff")
plt.plot(df["k"], 100*df["FDR_kplus"], marker="o", linewidth=2, label="Knockoff+")
plt.plot(df["k"], 100*df["FDR_BHq"], marker="^", linewidth=2, label="BHq")
plt.xlim(min(ks)-5, max(ks)+5)
plt.ylim(0, 25)
plt.xlabel("Sparsity level k")
plt.ylabel("FDR (%)")
plt.legend(loc="lower right", framealpha=0.9)
plt.tight_layout()
plt.show()
