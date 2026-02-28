# Unsupervised Anomaly Detection & Health Indicator (NASA C-MAPSS)

## Project overview
This project builds an end-to-end **condition-monitoring** pipeline for turbofan engines using multivariate time series from the **NASA C-MAPSS** benchmark. It produces two complementary outputs:

- **Early anomaly detection** without failure labels  
- A continuous, interpretable **Health Indicator (HI)** to summarize degradation over time

## Paper
A detailed write-up of the anomaly detection methodology and evaluation is included in this repository:

- `paper.pdf` — **Early Fault Detection on C-MAPSS with Unsupervised LSTM Autoencoders**


---
## 1) Unsupervised anomaly detection (sequence autoencoder)

**Goal:** learn *normal* operating behavior from healthy cycles and flag sustained deviations as early signals of degradation.

**Process (high level):**
- **Normalization:** operational settings and sensor signals are scaled with **MinMax** normalization to align magnitudes.
- **Healthy-only training split:** for each engine, the **first 85% of cycles** are treated as healthy and used for training.
- **Temporal windows:** the series is converted into **sliding windows (length = 10 cycles)** to capture dynamics and cross-sensor relationships.
- **Model:** an **LSTM autoencoder** (with an interchangeable **GRU** option) is trained to reconstruct healthy sequences by minimizing **MSE**.
- **Anomaly score:** the **reconstruction error (window MSE)** is used as the anomaly score.
- **Thresholding (automatic):** a statistical threshold is calibrated from the healthy training error distribution:

```math
\tau = \mu_{\mathrm{MSE}} + 2.5 \cdot \sigma_{\mathrm{MSE}}
```

Windows above $\tau$ are flagged as anomalous, enabling consistent comparison across engines and dataset splits.

---

## 2) Health Indicator (0–1)

**Goal:** transform reconstruction error into a smooth, operationally meaningful health signal where **1 ≈ healthy** and **0 ≈ failure-like**.

**Process (high level):**
- Reconstruction error is mapped into a normalized HI:

```math
\mathrm{HI} = 1 - \frac{\mathrm{MSE} - \mathrm{MSE}_{\min}}{\mathrm{MSE}_{\max} - \mathrm{MSE}_{\min}}
```

with clipping to $[0,1]$ for numerical stability.
- A visual **“optimal maintenance” band (0.15–0.35)** is overlaid to illustrate how HI can support decision-making through a practical intervention zone rather than relying only on binary alarms.

---

## What this demonstrates
- A **label-free predictive maintenance** workflow trained on healthy behavior only.
- Dual outputs: **binary anomaly flags** (alerts) + **continuous HI** (trending, prioritization, decision support).
- Interpretability and robustness via **explicit scoring + statistical thresholding**.

---

## Notebooks
- `anomaly_detection.ipynb`
- `health_indicator.ipynb`
