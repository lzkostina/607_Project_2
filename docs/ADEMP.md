# ADEMP: Simulation Framework for Knockoff and Knockoff+  
*Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*

---

## **Aims:** 
The goal of this project is to evaluate the empirical performance of the **Knockoff** and **Knockoff+**
procedures for variable selection in high-dimensional linear regression. Thus, in a sparse linear regression model
setting $y = X \beta + z$ we want to compare FDR control and power of Knockoff,
Knockoff+ and Benjamini–Hochberg methods. 

Specifically, we aim to answer the following questions:

1. **Does Knockoff+ control the False Discovery Rate (FDR) at the target level $q$?**
2. **How does its statistical power vary with:**
   - sparsity level $k$
   - signal amplitude $A$
   - correlation structure of the design matrix $X$
3. **How do Knockoff and Knockoff+ compare to each other?**  
4. **How do the results compare to classical multiple-testing methods $BH$?**

We replicated the experiments from **Section 3.3.1 (Equicorrelated / IID Gaussian) and
Section 3.3.2 (AR(1) correlation)** of Barber & Candès (2015),
using the same high-dimensional setting: $n = 3000, p = 1000$.

---

## **Data-Generating Mechanisms**

Each simulation trial draws data from the linear model: $y = X \beta + z$, where 
$ z \sim N(0, I_n).$

### **1. Design matrix $X$**

Two correlation regimes are included:
#### **(a) IID Gaussian**
- Mode: `"iid"`
- Each row: $X_i \sim N(0, I_p)$

#### **(b) AR(1) Gaussian**
- Mode: `"ar1"`
- Autocorrelation parameter: $ \rho \in \{0.5, 0.9\}$
- Covariance: $ \Sigma_{jk} = \rho^{|j-k|}$

### **2. Sample size and dimensionality**
- $ n = 3000 $ 
- $ p = 1000 $

### **3. Regression coefficients $\beta$**
A sparse vector of size  $p$:
- Number of signals: $ k \in \{10, 30, 60\}$
- Non-zero entries set to: $\beta_j = A$,  where signal amplitude $A \in \{2.5, 3.5, 5.0\}$

### **4. Noise**
- $z \sim N(0, I_n)$

### **5. Number of simulation trials**
- Per-condition: **n_trials = 600**
- For Figure 3.3.2 curves: **n_sim = 200** (for smoother power vs. k plots)
---

## **Estimands / Targets**

The simulation focuses on two performance measures:
1. **False Discovery Rate (FDR)**  
   $$
   \mathrm{FDR} = \mathbb{E}\left[\frac{V}{R \vee 1}\right]
   $$

2. **Power**  
   $$
   \text{Power} = \frac{\text{# true positives}}{k}
   $$
---

## **Methods**

The following methods were evaluated:

### **1. Knockoff**  
- Generates equicorrelated knockoffs $\widetilde X$
- Statistic: Lasso path-based signed max statistic
- Selection rule: standard Knockoff threshold

### **2. Knockoff+**  
- Same knockoff generation  
- Statistic identical  
- Threshold uses augmented denominator:
  $$
  \widehat{\mathrm{FDP}}^+ = \frac{1 + \#\{ j : W_j \le -t \}}{\#\{ j : W_j \ge t \}}
  $$

### **3. Benjamini–Hochberg (BHq)**  
- Used as a baseline in the code  

---

## **Performance measures**

What metrics will you use to evaluate performance? (e.g., bias, variance, MSE, coverage, power, Type I
error)
Include a table summarizing your simulation design matrix showing all combinations of conditions. Write a description of the simulation
that would be sufficient for someone else to reproduce it.

Each method is evaluated using the following metrics:

1. **False Discovery Rate (FDR)**
2. **Power**

For each condition, results are aggregated across trials:

$$
\widehat{\mathrm{FDR}}=\frac{1}{T}\sum_{t=1}^T \mathrm{FDP}_t,
\qquad
\widehat{\mathrm{Power}}=\frac{1}{T}\sum_{t=1}^T \mathrm{Power}_t.
$$

---


## **Summary of Simulation Design Matrix**

All combinations of:

| Parameter      | Values        |
|----------------|---------------|
| Mode           | iid, ar1      |
| $\rho$         | 0.0, 0.5, 0.9 |
| $k$            | 10, 30, 60    |
| $A$            | 2.5, 3.5, 5.0 |
| $n$            | 3000          |
| $p$            | 1000          |
| Trials         | 600           |
| FDR target $q$ | 0.2           |

---
