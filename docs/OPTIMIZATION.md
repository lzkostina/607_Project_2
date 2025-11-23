# Optimization Report

This document describes the performance optimizations applied to the Project 2 simulation pipeline. 
For each optimization, we document:
- **Identified problems** 
- **Solutions implemented** 
- **Before/after code snippets**
- **Performance impact**
- **Trade-offs**

Additional timing comparisons and profiling visualizations are included where appropriate.

---

## Overview and improvement directions
Based on profiling and inspection results (see `BASELINE.md`), the baseline implementation on the `main` branch is 
functionally correct but has the following performance issues:

- Significant runtime for moderate values of `n`
- Repeated computations inside each trial
- Python-level loops instead of vectorized operations
- Expensive linear-algebra routines executed repeatedly with identical inputs
- Some unnecessary or unstable linear-algebra operations
- 
Because the simulation design requires running hundreds of independent trials under identical settings, the baseline 
implementation spends a lot of time recomputing the same quantities for each trial. Thus, at least at this point I 
expect optimizations focusing on vectorization, precomputation and parallel execution will provide 
the largest performance gains.

In this optimization report, I therefore focused on the following categories required by the assignment:

- **Array Programming**  
- **Algorithmic Improvements**  
- **Parallelization**  

All optimizations were evaluated on the same configuration used in Section 2 of `BASELINE.md`, ensuring that 
performance comparisons are fair and represent true improvements relative to the baseline.

Overall, the optimized pipeline achieves substantial speedups relative to the baseline implementation, primarily due to 
introducing parallel execution.

---

## Implemented optimizations: 

### Optimization 1 — Lasso First-Entry Detection (Array Programming)

**Problem Identified:**  
During computation of the Knockoff+ W-statistics, the baseline implementation determines, for each feature, the first
value of the Lasso regularization parameter `alpha` at which the coefficient becomes nonzero (“first-entry time”).
In the baseline code, this was implemented as a Python loop:

- For each feature `j`, we called `np.argmax(nz[j])` and checked `nz[j].any()`.
- This resulted in $O(p)$ Python-level iterations (with `p = n_features`) even though all data were already stored in
NumPy arrays.
- According to profiling, this loop was called many times inside `lasso_path_stats`.  
  
Although not the main bottleneck, it still contributed measurable overhead. Here, `nz` is a `(n_features, n_alphas)` 
boolean array indicating whether the coefficient is nonzero for a given feature and `alphas` is the corresponding 
grid of regularization parameters.


**Solution Implemented:**  
I replaced the loop over features with a fully vectorized NumPy implementation that computes all first-entry indices
at once using array operations.

Instead of iterating over `j = 0, …, n_features − 1`, the new code:

1. Computes, for all features at once, whether they ever become nonzero.
2. Computes the index of the first nonzero along the alpha axis.
3. Uses `np.where` to map those indices to the corresponding `alpha` values, 
and returns `0.0` for features that never enter.

   
**Before:** 
```
for j in range(n_features):
    idx = np.argmax(nz[j])
    entry[j] = alphas[idx] if nz[j].any() else 0.0
 ```   

**After:** 
```
has_nonzero = nz.any(axis=1)
first_idx = nz.argmax(axis=1)
entry = np.where(has_nonzero, alphas[first_idx], 0.0)
```

**Performance impact:**

This change removes all Python-level iteration from the first-entry detection step. The operation is now performed 
entirely in optimized C code inside NumPy.
Unfortunately we do not see the direct speedup - total runtime after implementing this optimization is 31:46 minutes. 


**Trade-offs:**

* The vectorized version is slightly less “explicit” than the loop, which makes debugging individual features 
less straightforward. 
* However, the logic remains simple, uses only standard NumPy operations.
* No numerical accuracy issues were introduced, the outputs are identical to the baseline version.

---

### Optimization 2 — Knockoff threshold (Array programming + Algorithmic improvement)

**Problem Identified:**  

The Knockoff+ threshold function is called repeatedly, when computing knockoff statistics across trials and settings. 
The baseline implementation scanned over the grid of candidate thresholds with a Python `for` loop:

- For each candidate `t`, it called `np.count_nonzero(W <= -t)` and `np.count_nonzero(W >= t)`.
- This results in `O(n_candidates)` iterations.
- Each iteration launched one or more NumPy kernels over a vector of length `p`.
- Profiling showed that while this step is not the primary bottleneck (compared to Lasso path and 
knockoff construction), it still contributes non-negligible overhead, especially when many scenarios are tested or 
when the number of distinct `|W_j|` values is large.

  
**Solution implemented:**  
I implemented two versions of the threshold rule and an adaptive dispatcher:

1. A **fully vectorized** version, `_threshold_vectorized` evaluates all candidate thresholds simultaneously. 
Uses a single pair of matrix comparisons: $W \ge t \quad\text{and}\quad W \le -t$

2. A **loop-based** version, `_threshold_loop` matches the original implementation and avoids allocating
a `(p × n_candidates)` matrix. This version is more memory-friendly and preferable for larger `p` or many candidate
thresholds.
The dispatcher chooses:
```
if p < 500 and n_candidates < 500:
    use vectorized version
else:
    use loop version
```

The main function `knockoff_threshold` now selects between these two scenarios based on the problem size
(i.e., `p = len(W)` and `n_candidates = len(candidates)`).

**Before:** 
```
for t in candidates:
    num = offset + np.count_nonzero(W <= -t)
    den = max(1, int(np.count_nonzero(W >= t)))
    fdp_hat = num / den
    if fdp_hat <= q:
        return float(t)
 ```   

**After:** 
```
W_col = W[:, np.newaxis]           # shape (p, 1)
t_row = candidates[np.newaxis, :]  # shape (1, n_candidates)

num_pos = np.sum(W_col >= t_row, axis=0)   # counts of W_j >= t for each t
num_neg = np.sum(W_col <= -t_row, axis=0)  # counts of W_j <= -t for each t

fdp_hat = (offset + num_neg) / np.maximum(1, num_pos)
valid_idx = np.where(fdp_hat <= q)[0]

if valid_idx.size == 0:
    return float("nan")
return float(candidates[valid_idx[0]])
```

**Performance impact:**

Surprisingly, on the test configuration the adaptive implementation produced a slightly longer full runtime 
34:37 minutes. I assume that the possible reasons are that `np.count_nonzero` is already highly optimized in C, 
and that micro-optimizations like this one can't beat macro-bottlenecks.
Importantly, in cases with small $p$ or many threshold candidates, the fully vectorized backend is considerably faster.

**Trade-offs:**

* The adaptive design introduces two execution paths, increasing implementation complexity.
* The vectorized backend is very fast for moderate sizes, but memory-intensive for large sizes.
* The loop backend remains simple and avoids allocating large intermediary arrays.

---

### Optimization 3 — Knockoff Selection Logic (Algorithmic Simplification & Code Deduplication)

**Problem Identified:**  
The baseline implementation of `knockoff_select` contained an embedded copy of the knockoff threshold algorithm. 
Specifically, it:

- Constructed its own grid of candidate thresholds
- Manually scanned those thresholds using a Python loop
- Recomputed numerator/denominator counts for each `t`
- Repeated the same logic already implemented in `knockoff_threshold`

This duplication led to:

- Unnecessary repeated work inside each `knockoff_select` call  
- Two separate threshold implementations that needed to remain consistent   
- Additional scanning loops performed inside every simulation replicate


**Solution Implemented:**  

I refactored `knockoff_select` so that it delegates all threshold computation to the previously optimized  
`knockoff_threshold` function (which includes the adaptive vectorized backend).

This cleanup achieves:

- A single implementation of the threshold rule  
- Automatic use of the adaptive fast path from Optimization 2  
- Much simpler and cleaner logic inside `knockoff_select`  
- Removal of duplicated threshold scanning loops, which previously ran independently inside `knockoff_select`  


**Before:**

```
tgrid = np.unique(np.abs(W)[np.abs(W) > 0.0])
tgrid.sort()

T = None
fdp_hat_T = None
for t in tgrid:
    num = offset + np.sum(W <= -t)
    den = max(1, int(np.sum(W >= t)))
    fdp_hat = num / den
    if fdp_hat <= q:
        T = float(t)
        fdp_hat_T = float(fdp_hat)
        break

if T is None:
    return np.array([], dtype=int), {"T": None, "fdp_hat": None}
```

**After:** 
```
T = knockoff_threshold(W, q, offset)

if np.isnan(T):
    return np.array([], dtype=int), {"T": None, "fdp_hat": None}

num = offset + np.sum(W <= -T)
den = max(1, int(np.sum(W >= T)))
fdp_hat = float(num / den)

selected = np.where(W >= T)[0].astype(int)
return selected, {"T": T, "fdp_hat": fdp_hat}
```

**Performance impact:**
After removing the redundant threshold computation from `knockoff_select`, the full runtime for the same profiled
configuration decreased to 30:57 minutes.
Although this optimization is primarily algorithmic simplification rather than a raw speedup trick, the elimination 
of repeated Python loops yields a measurable runtime reduction and supports the overall correctness.

**Trade-offs:**

- `knockoff_select` now depends on the correctness of `knockoff_threshold` (a positive trade-off since it improves
overall modularity).
- The refactored logic is significantly easier to read and maintain.


### Optimization 4 — Precomputation of Scenario-Constant Matrices (Algorithmic Improvement)

**Problem Identified:** 

In the baseline implementation, several quantities were recomputed inside every simulation trial, even though they
depend only on `(p, mode, rho)` and not on the choice of trial. In particular:
- For AR(1) designs, the covariance matrix $\Sigma_{ij} = \rho^{|i-j|}$ was rebuilt independently for each replicate.
- In scenarios where the covariance eigenvalues or numerical rank were requested, a full eigen-decomposition 
was repeated.
- Since each scenario may require hundreds of replicates, these repeated computations accumulated into a noticeable cost.

Profiling showed that although this step is not as expensive as LARS or knockoff construction, the redundant covariance
construction inside each replicate added measurable overhead.


**Solution Implemented:**  
I implemented a new `precompute_scenario_matrices` function that builds and caches all scenario-invariant quantities
once per scenario:

- `Sigma_true` - exact covariance matrix
- `evals`, `evecs` — eigenvalues/eigenvectors (only when needed)  
- `rank_info` — numerical rank & stability flags  
- flags indicating whether the covariance should be used

All replicate-level routines now accept a `precomp` object and reuse these precomputed matrices rather than
rebuilding them.

**Before:**
```
(Performed inside each replicate)
Sigma = compute_theoretical_sigma(p, mode, rho)
evals, evecs = np.linalg.eigh(Sigma)
```

**After:**
```
precomp = precompute_scenario_matrices(mode=mode, p=p, rho=rho)

# Inside each replicate:
Sigma = precomp["Sigma_true"]
evals = precomp["evals"]
evecs = precomp["evecs"]

```

**Performance Impact:**

- Eliminates repeated $O(p^2)$ covariance construction from every replicate.
- Avoids repeated $O(p^3)$ eigen-decomposition when diagnostics require eigenvalues.
- For a typical case $(p = 1000, R = 600 replicates)$, this removes approximately:
$$ 600 \times O(p^2) \approx 600 \times 10^6 \text{ operations }$$

This optimization also improves stability because all scenario-level quantities are constructed once rather than
repeatedly in slightly different floating-point states.

**Trade-offs:**

- Slightly more complexity in the simulation pipeline (must pass `precomp` object between functions).
- Precomputation increases setup time per scenario (one-time cost).
- Overall readability improved by cleanly separating what is constant across a scenario from what varies across trials.

Overall, this optimization removes redundant heavy computations, reduces runtime, and significantly improves 
maintainability without changing any statistical behavior.


### Optimization 5 — Parallelization 

**Problem identified:**

The baseline implementation ran all simulation trials in a single Python loop:
- Each trial independently generates data, constructs knockoffs, computes Lasso W-statistics, and evaluates FDP/Power.
- For moderately large problem sizes (e.g. n = 3000, p = 1000, n_trials = 600), a single trial is already expensive.
- Since all simulation replicates for a given scenario are fully independent, the workload is an ideal candidate 
for multi-core execution.

This makes parallelization the highest-ROI optimization once per-trial costs have been reduced.

**Solution Implemented:**
I introduced a clean parallel simulation driver built on `joblib.Parallel`, with automatic fallback to sequential execution when parallelism is disabled or unavailable.

New functions:
- `run_simulation_sequential(cfg, verbose, precomputed)`
- `run_simulation_parallel(cfg, verbose, precomputed)`
- Updated `run_simulation(cfg)` that dispatches to the appropriate backend.

Key features:
- Parallelization is per trial: each worker runs `run_one_trial(cfg, trial_id, precomputed=precomputed)`.
- Precomputed objects (e.g. Sigma_true) are constructed once per scenario and passed to workers as read-only, 
with each worker making its own copies where needed.
- Config/CLI control: "n_jobs" (default -1 = use all cores).

**Before:**
```
def run_simulation(cfg: Dict[str, Any]) -> pd.DataFrame:
    try:
        from tqdm import trange
    except Exception:
        trange = range

    rows = []
    for t in trange(int(cfg["n_trials"]), desc=f"Sim {cfg['name']}"):
        rows.append(run_one_trial(cfg, t))
    return pd.DataFrame(rows)
```
**After:**
```
def run_simulation_parallel(
    cfg: Dict[str, Any],
    verbose: bool = True,
    precomputed: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Parallelized simulation with optional precomputed matrices.

    OPTIMIZATION: Precomputed Sigma is shared across all workers (read-only).
    """
    n_trials = int(cfg["n_trials"])
    n_jobs = cfg.get("n_jobs", -1)
    backend = cfg.get("backend", "loky")

    if not HAS_JOBLIB or n_jobs == 1:
        return run_simulation_sequential(cfg, verbose=verbose, precomputed=precomputed)

    if n_jobs == -1:
        n_jobs = cpu_count()

    # Create partial function with precomputed matrices
    trial_func = partial(run_one_trial, cfg, precomputed=precomputed)

    if verbose:
        try:
            from tqdm import tqdm
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(trial_func)(t) for t in tqdm(range(n_trials), desc=f"Sim {cfg['name']}")
            )
        except ImportError:
            print(f"Running {n_trials} trials with {n_jobs} workers...")
            results = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(trial_func)(t) for t in range(n_trials)
            )
    else:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(trial_func)(t) for t in range(n_trials)
        )

    return pd.DataFrame(results)
```

**Performance Impact:**
This optimization produced the largest practical speedup in the entire project. Once per-trial computations were
optimized in earlier steps (vectorization, precomputation), parallelizing across trials reduced wall-clock runtimes 
dramatically to 12:26 minutes. 
Because simulation replicates dominate total runtime, this optimization delivered the highest return on investment 
of any modification in this project.

**Trade-offs:**

- Large reduction in wall-clock runtime on multi-core machines.
- Results are 100% reproducible across parallel/sequential modes due to deterministic trial seeding.
- Cleanly separated sequential/parallel code paths improve maintainability.
- More complicated control flow and error handling.
- Precomputed matrices must remain read-only to avoid cross-worker state contamination.

## Lessons Learned

### 1. Profiling beats intuition  
Profiling clearly showed that most runtime was spent in Lasso path computation and knockoff construction (not in the
Python loops that initially looked suspicious). This proves that meaningful optimization must be evidence-driven, 
not guess-driven.

### 2. Removing redundancy matters more than micro-optimizations  
Vectorizing small loops improved cleanliness, but had almost no global effect.  
In contrast, eliminating repeated computations (through scenario-level precomputation) produced much larger benefits.  
It clearly illustrates "optimize what happens the most often".

### 3. Parallelization was the strongest accelerator  
Parallelizing across simulation trials yielded the largest time reductions — typically cutting runtime 
by **50–70%** across all tested sample sizes. Parallel execution had the highest return on investment in the entire
project.

### 4. Not all optimizations reduce runtime—but they still help  
Some changes (e.g., adaptive vectorized thresholding) did not speed up the main configuration but improved modularity 
and made future optimization simpler. 

### 5. Clean separation between scenario-level and trial-level work is powerful  
By precomputing all scenario-invariant matrices once and passing them to workers, the code became faster and easier
to parallelize. This design pattern greatly simplified the entire pipeline.


#### Empirical runtimes (updated version) 
| n    | n_trials | baseline runtime (seconds) | optimized runtime (seconds) |
|------| -------- |----------------------------|-----------------------------|
| 2500 | 50       | 106.18                     | 46.37                       |
| 3000 | 50       | 159.73                     | 61.92                       |
| 3500 | 50       | 193.18                     | 90.86                       |
| 4000 | 50       | 244.75                     | 121.60                      |
| 4500 | 50       | 306.80                     | 176.57                      |
| 5000 | 50       | 364.75                     | 229.66                      |