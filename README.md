# Stochastic Systems, Predictive Estimation & Spectral Analysis
### A Three-Project Computational Research Portfolio

> *This portfolio is part of a broader study on stochastic systems, predictive estimation, and data analytics using computational simulation techniques.*

---

## Overview

This repository contains three self-contained simulation projects that together form a unified study of stochastic processes from three complementary analytical perspectives:

| Project | Topic | Domain |
|---------|-------|--------|
| **P1** | Scaled Symmetric Random Walk | Financial Market Modeling |
| **P2** | Kalman Filter for AR(1) Signals | Forecasting & Predictive Analytics |
| **P3** | Smoothed Periodogram Estimation | Frequency & Data Analytics |

Each project is independently complete with its own theory, simulation, and results — but together they demonstrate how stochastic modeling, estimation theory, and spectral analysis contribute to modern computational analytics.

---

## Repository Structure

```
├── P1_Random_Walk.py          ← Project 1 simulation script
├── P2_Kalman_Filter.py        ← Project 2 simulation script
├── P3_Periodogram.py          ← Project 3 simulation script
│
├── P1_Random_Walk_Report.docx ← Full academic report (P1)
├── P2_Kalman_Filter_Report.docx ← Full academic report (P2)
├── P3_Periodogram_Report.docx ← Full academic report (P3)
│
└── README.md                  ← This file
```

---

## Requirements

All three scripts require only standard Python scientific libraries.

```bash
pip install numpy matplotlib
```

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.21 | Numerical computation, random number generation |
| `matplotlib` | ≥ 3.4 | Plotting and visualization |

> **Python version:** 3.7 or higher recommended.

---

## How to Run

### Option A — Run individually in VS Code
Open any `.py` file in VS Code and press the **▶ Run** button (top-right corner).

### Option B — Run from terminal
```bash
python P1_Random_Walk.py
python P2_Kalman_Filter.py
python P3_Periodogram.py
```

Each script will:
1. Run the simulation
2. Display the plot in a window
3. Save the figure as a `.png` file in the same directory

---

## Project Summaries

---

### P1 — Scaled Symmetric Random Walk
**File:** `P1_Random_Walk.py`
**Output:** `P1_Random_Walk.png`

**What it does:**
Simulates 100 sample paths of the scaled symmetric random walk:

```
W_N(t) = (1/√N) · Σ X_i    for t ∈ [0, 1]
```

where each X_i = ±1 with probability p = 0.5, and N = 100.

**Parameters:**
```python
N       = 100     # number of steps
n_paths = 100     # number of simulated paths
p       = 0.5     # probability of +1 step
seed    = 42      # random seed for reproducibility
```

**Plots generated:**
- Five sample paths overlaid with the mean function
- Estimated mean function E[W_N(t)]
- Empirical covariance function at fixed t = 0.5

**Key results:**
- Mean function converges to zero (zero-drift confirmed)
- Empirical covariance matches theoretical C(s,t) = min(s,t)
- Paths exhibit Brownian motion-like irregular behavior

**Theory connection:** By Donsker's invariance principle, W_N(t) converges to standard Brownian motion as N → ∞. This forms the basis of stock price modeling and Black-Scholes theory.

---

### P2 — Kalman Filter for AR(1) Signal
**File:** `P2_Kalman_Filter.py`
**Output:** `P2_Kalman_Filter.png`

**What it does:**
Implements the Kalman filter to estimate a hidden AR(1) signal from noisy observations.

```
State:       Z_n = 0.8·Z_{n-1} + W_n     W_n ~ N(0, 0.36)
Observation: X_n = Z_n + N_n             N_n ~ N(0, 1.0)
```

**Parameters:**
```python
a       = 0.8    # AR(1) coefficient
Q       = 0.36   # system noise variance  (std dev = 0.6)
R       = 1.0    # observation noise variance (std dev = 1.0)
n_steps = 100    # number of time steps
seed    = 7      # random seed for reproducibility
```

**Kalman filter equations:**
```
Predict:  Z̃_n = a·Ẑ_{n-1}           P̃_n = a²·P_{n-1} + Q
Update:   K_n  = P̃_n/(P̃_n + R)      Ẑ_n  = Z̃_n + K_n·(X_n - Z̃_n)
                                       P_n  = (1 - K_n)·P̃_n
```

**Plots generated:**
- True signal Z_n vs noisy observation X_n vs Kalman estimate Ẑ_n
- Estimation error with ±1 standard deviation bounds (from P_n)
- Kalman gain convergence K_n → steady-state value

**Key results:**
- Kalman gain converges to steady-state ≈ 0.26 within a few steps
- Estimation error is zero-mean and bounded by √P_n
- Filter achieves MMSE-optimal reconstruction under Gaussian assumptions

**Theory connection:** The Kalman filter is the optimal linear estimator under linear-Gaussian assumptions. It is widely used in GPS, financial forecasting, autonomous vehicles, and sensor fusion.

---

### P3 — Smoothed Periodogram (Spectral Density Estimation)
**File:** `P3_Periodogram.py`
**Output:** `P3_Periodogram.png`

**What it does:**
Estimates the power spectral density (PSD) of an i.i.d. Uniform(0,1) sequence by averaging M independent periodograms (Bartlett's method).

```
Periodogram:         Î_m(f) = (1/N)|FFT(X_n - X̄)|²
Smoothed estimate:   Ŝ(f)   = (1/M) · Σ Î_m(f)
True PSD:            S_X(f) = 1/12 ≈ 0.0833  (flat white noise)
```

**Parameters:**
```python
N      = 256          # samples per periodogram segment
M_vals = [10, 20, 50] # number of averaged periodograms
seed   = 99           # random seed for reproducibility
```

**Plots generated:**
- Smoothed periodogram for M = 10 (high variance)
- Smoothed periodogram for M = 20 (moderate variance)
- Smoothed periodogram for M = 50 (low variance)
- True PSD = 1/12 overlaid on all three panels (log scale)

**Key results:**
- Variance reduces as 1/M — doubling M halves spectral variance
- At M = 50, smoothed estimate closely tracks the true flat PSD
- Mean-centering correctly removes the DC component from U(0,1) mean = 0.5

**Theory connection:** Spectral estimation is fundamental to EEG analysis, vibration monitoring, communication systems, and economic cycle detection.

---

## Theoretical Connections Across Projects

```
P1 (Random Walk)
    └─ Discrete stochastic process → converges to Brownian motion
    └─ Foundation of: Black-Scholes, Monte Carlo finance

P2 (Kalman Filter)
    └─ State-space model → recursive Bayesian estimation
    └─ Foundation of: GPS, forecasting, sensor fusion

P3 (Periodogram)
    └─ Frequency-domain analysis → spectral density estimation
    └─ Foundation of: EEG, communications, vibration analytics

All three → Stochastic Modeling + Estimation Theory + Data Analytics
```

---

## Output Files

After running all three scripts, the following PNG figures are saved:

| File | Description |
|------|-------------|
| `P1_Random_Walk.png` | Three-panel plot: sample paths, mean, covariance |
| `P2_Kalman_Filter.png` | Three-panel plot: signal tracking, error bounds, Kalman gain |
| `P3_Periodogram.png` | Three-panel plot: smoothed PSD for M = 10, 20, 50 |

---

## Reproducibility

All three scripts use fixed random seeds for full reproducibility:

```python
# P1
np.random.seed(42)

# P2
np.random.seed(7)

# P3
np.random.seed(99)
```

Running any script multiple times will always produce the same figures.

---

## Academic Context

Each project report follows a consistent 11-section structure:

1. Introduction
2. Objective
3. Theoretical Background
4. Mathematical Model
5. Algorithm / Methodology
6. Code Implementation
7. Simulation Results
8. Interpretation
9. Industry Applications
10. Conclusion
11. Broader Research Perspective

The full reports are available in the `.docx` files included in this repository.

---

## Tools Used

| Tool | Role |
|------|------|
| Python | Simulation and numerical computation |
| NumPy | Array operations, random number generation, FFT |
| Matplotlib | Visualization and figure export |

---

*Together, these projects demonstrate how stochastic modeling, estimation theory, and spectral analysis contribute to modern computational analytics — spanning financial modeling, predictive systems, and frequency-domain analysis.*
