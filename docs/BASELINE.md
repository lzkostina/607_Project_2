# Baseline Performance 
This file summarizes performance of the Unit 2 project before applying any optimizations.  
It has the following structure:

* Total runtime of the full simulation study
* Runtime profiling results (bottlenecks)
* Computational complexity analysis 
* Numerical warnings / instability evidence
---

## 0. Environment & Baseline Setup

**Commit / tag used:**  
`unit2-baseline`  

**Machine:**  
- CPU: `M1`  
- RAM: `16GB`  
- OS: `macOS Sequoia 15.7.1`  
- Python version: `Python 3.12.6`  

The full experiment consists of recreating Figure 1 and Figure 2 from the original article; these two plots are 
computationally cheap. The main computational burden comes from Section 3.3.2, which investigates the effect of 
sparsity level $k$, signal amplitude $A$, and correlation $\rho$:

- *Sparsity level:* $k = 10, 20, 30, \dots, 200$  
- *Signal amplitude:* $ = 2.8, 2.9, \dots, 4.2$  
- *Correlation:* $\rho = 0, 0.1, \dots, 0.9$

This corresponds to $20 + 15 + 10 = 45$ different settings.  
**Command used to run full simulation:**  
```bash
make all
```
Each individual setting requires approximately 30 minutes, and running all settings locally would take a little less 
than a day. Since the computational cost is primarily determined by $(n, p, n\_\text{trials})$ rather than by $(k, A)$, 
I selected one representative configuration for detailed profiling.

---

## 1. Runtime of Simulation Study
This section measures the wall-clock time of simulation for one set of parameters, run exactly as originally submitted.

Since the cost of each setting is dominated by n, p, and n_trials rather than k or A, I selected one representative 
configuration for profiling.I decided to study one of the mentioned settings more closely 
(not to wait each time for a day if running locally). 
**Command executed:**
```bash
PYTHONPATH=. python src/simulation.py --simulate -c configs/ar1_rho0p5_p1000_n3000_k10_A3p5.json
````
Summary:
- Total baseline runtime: $\approx 23$ hours
- Runtime for one set of parameters 30:06 minutes

## 2. Runtime Profiling (One Representative Parameter Set)
I restricted baseline simulation for a single representative configuration. Config used:
configs/ar1_rho0p5_p1000_n3000_k10_A3p5.json, using the same code path as the Unit 2 study.
* $n = 3000$
* $p = 1000$
* $k = 30$
* $A = 3.5$
* $\rho = 0.5$
* mode $=$"ar1"
* n_trials $= 600$
* $q = 0.2$

**Command used to run baseline simulation:**  
```bash
 PYTHONPATH=. python -m cProfile -o results/profiling/prof_ar1_rho0p5_p1000_n3000_k10_A3p5.prof \
    src/simulation.py --simulate -c configs/ar1_rho0p5_p1000_n3000_k10_A3p5.json
```
```bash
snakeviz results/profiling/prof_ar1_rho0p5_p1000_n3000_k10_A3p5.prof
```
### Top bottlenecks (cumulative time)

| Rank | Function / Location                         | Description / Role                                                                                                         |
|------|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| 1    | `_lars_path_solver` (`_least_angle.py:411`) | Core LARS/Lasso-path solver called from `lasso_path_stats`. Dominant low-level routine.                                     |
| 2    | `qr` (`_linalg.py:986`)                     | QR factorization repeatedly called inside Lasso path and knockoff construction.                                              |
| 3    | `eigh` (`_linalg.py:1536`)                  | Eigenvalue decomposition used by equi-correlated knockoff construction.                                                      |
| 4    | `knockoffs_equicorr` (`methods.py:8`)       | High-level knockoff construction wrapper; dominates cumulative time because it calls `qr`, `eigh`, `cholesky`, etc.          |
| 5    | `lasso_path_stats` (`methods.py:127`)       | Computes Knockoff+ W-statistics along the Lasso path; repeatedly triggers `_lars_path_solver`.                               |
| 6    | `generate_design` (`dgps.py:36`)            | DGP routine that constructs the design matrix; moderately expensive compared to knockoff/Lasso components.                   |
| 7    | `bh_select_whitened` (`methods.py:357`)     | BH on whitened z-scores; involves covariance whitening steps.                                                                |
| 8    | `solve` (`_linalg.py:382`)                  | Linear system solver used inside both knockoff construction and whitened BH.                                                 |
| 9    | `bh_select_marginal` (`methods.py:278`)     | Baseline BH using marginal z-scores; lightweight relative to Knockoff+.                                                      |
| 10   | `_solve_triangular` (`_basic.py:512`)       | Low-level triangular-solver inside knockoff and Lasso computations.                                                           |

A few notable lower-level helpers also appear near the top:
* _lars_path_solver is by far the biggest individual function.
* QR and eigenvalue decomposition are substantial and called many times.
* Knockoff construction (knockoffs_equicorr) has a large cumulative time but smaller direct time because it delegates to heavy linear algebra routines.
* Lasso path (lasso_path_stats) has moderate direct time but high cumulative due to solver calls.


### Interpretation of profiler output

**Overall structure.**  
The top-level function `run_one_trial` indirectly accounts for most of the runtime because it calls data generation, 
knockoff construction, Lasso path computation, and multiple BH-based procedures for each of the 600 trials.

**Knockoff + Lasso path dominate.**  
The largest contributors are:

- `lasso_path_stats` → `_lars_path_solver`  
- `knockoffs_equicorr` → `qr`, `eigh`, `cholesky`, `_solve_triangular`  

These routines together consume the majority of cumulative time.  
This confirms that the **Knockoff+ pipeline is the dominant bottleneck**, not BH-based baselines or the DGP.

**Linear algebra is the core cost.**  
The fact that `qr`, `eigh`, `solve`, and `_solve_triangular` are all in the top 10 shows that expensive matrix 
operations, repeated for each trial, drive runtime.

**Data generation costs are moderate.**  
`generate_design` and the DGP-related helpers (`auto_regressive_cov`, `scale_cols_unit_l2`) appear in the profile but 
are an order of magnitude cheaper than Knockoff+.

**BH-based methods are very cheap.**  
`bh_select_marginal` and `bh_select_whitened` contribute little to the overall runtime.  
This indicates that Knockoff+ is the only computationally expensive method in the study.

## 3. Computational Complexity Analysis

To empirically assess computational complexity, I varied the sample size $n$ while keeping
all other parameters fixed to match the profiled AR(1) scenario:

- mode = "ar1", $\rho = 0.5$  
- $p = 1000, k = 10, A = 3.5, q = 0.2$  
- n_trials (for this experiment) $= 50$  

I used the baseline implementation (`run_simulation`) and measured wall-clock runtime
for each value of $n$ using `time.perf_counter()`.

Command:

```bash
PYTHONPATH=. python scripts/complexity_baseline.py
```
This script writes results/profiling/complexity_baseline.csv.

#### Empirical runtimes
| n    | n_trials | runtime (seconds) |
|------| -------- |-------------------|
| 2500 | 50       | 106.18            |
| 3000 | 50       | 159.73            |
| 3500 | 50       | 193.18            |
| 4000 | 50       | 244.75            |
| 4500 | 50       | 306.80            |
| 5000 | 50       | 364.75            |

I also created a log–log plot of runtime vs $n$ to inspect results visually 
File: results/figures/complexity_baseline.png

### Interpretation
The empirical runtimes increase faster than linear growth with sample size.  
A log–log fit suggests:

$\text{runtime} \approx O(n^{1.74})$

This matches expectations from the algorithm:

- Knockoff construction repeatedly performs eigenvalue decompositions and QR factorizations.
- The Lasso path computation calls `_lars_path_solver` many times, each involving costly matrix operations.


Thus, results indicate that optimizations targeting linear-algebra calls (caching, batching, or parallelizing trials)  
are likely to give the largest performance improvements.
