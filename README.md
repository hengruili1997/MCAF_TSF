# MCAF: Multi-domain Collaborative Analytics Framework for Time Series Forecasting

This repository provides the official PyTorch implementation of **MCAF** (Multi-domain Collaborative Analytics Framework).

MCAF is a **channel-independent** model for multivariate time series forecasting. It adopts a hierarchical ternary decoupling strategy to decompose time series into **Trend**, **Seasonal**, and **Residual** components. Each component is modeled in its corresponding domain (Time, Frequency, and Time-Frequency), with the goal of improving predictive performance and robustness.

> **Key Idea:** To better address non-stationary disturbances that often challenge conventional approaches, MCAF incorporates **DecompMAE** (a Time-Frequency Masked Autoencoder) together with a **Complex Fusion** mechanism, enabling effective integration of multi-domain representations.


---

## 🚀 Method Overview

### 1. Main Architecture

MCAF follows a "Decompose - Model - Fuse" paradigm:

1. **Signal Decomposition**:
* **Adaptive Trend Extraction**: Uses multi-scale convolution and entropy-aware dynamic weighting to extract long-term trends in the **Time Domain**.
* **DecompMAE**: Applies STFT and a Masked Autoencoder to disentangle periodic **Seasonal** patterns from irregular **Residual** noise in the **Time-Frequency Domain**.


2. **Multi-domain Collaborative Modeling**:
* **Trend (Time Domain)**: Modeled via an LSTM conditioned on adaptive basis weights.
* **Seasonal (Frequency Domain)**: Enhanced via Top-K sparse frequency amplification.
* **Residual (Time-Frequency Domain)**: Captured using Dictionary Learning or Convolutional Projection to handle transient anomalies.


3. **Complex Fusion**:
* Projects all components into a unified **Complex-valued Representation Space** and fuses them using a multi-head attention mechanism.



*(Placeholder for your architecture diagram)*

### 2. DecompMAE Module

The **DecompMAE** module is the core of our decomposition strategy. It applies Short-Time Fourier Transform (STFT) to the detrended signal and uses a lightweight reconstruction network to recover masked seasonal harmonics, effectively isolating non-stationary residuals.



## 🛠 Prerequisites

* Python 3.7+
* PyTorch >= 1.8.0 (Required for `torch.fft` and `torch.cfloat` support)
* numpy
* pandas
* scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt

```

---

## 💾 Datasets

We follow the standard data format used in Time Series Library (TSlib). You can download standard benchmarks (ETTh1, ETTm1, Traffic, Electricity, etc.) from:

* [Google Drive](https://drive.google.com/drive/folders/1QPM7MMKlzVffdzbGGkzARDuIqiYRed_f?usp=drive_link)
* [NJU Box](https://box.nju.edu.cn/d/abc2bbd7cff6461eb4da/)

Place the datasets in the `dataset/` folder:

```
dataset/
  ├── ETT-small/
  ├── traffic/
  ├── electricity/
  └── ...

```

---

## 🏃 Usage

### Run on Standard Datasets

To replicate our results on standard benchmarks (e.g., ETTm1), run the scripts provided in `scripts/`:

```bash
# Example: Running MCAF on ETTm1
sh scripts/long_term_forecast/ETT_script/MCAF_ETTm1.sh

```




## 🔗 Acknowledgement

This code is built upon the excellent work of the following repositories:

* [SOFTS](https://github.com/Secilia-Cxy/SOFTS) (Base structure)
* [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
* [iTransformer](https://github.com/thuml/iTransformer)

We appreciate their contribution to the time series forecasting community.

