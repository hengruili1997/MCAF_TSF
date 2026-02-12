# MCAF: Multi-domain Collaborative Analytics Framework for Time Series Forecasting

The official PyTorch implementation for **MCAF** (Multi-domain Collaborative Analytics Framework).

MCAF is a novel **Channel-Independent** multivariate time series forecasting model that leverages a hierarchical ternary decoupling mechanism. It decomposes time series into **Trend**, **Seasonal**, and **Residual** components and models them in their native domains (Time, Frequency, and Time-Frequency) to achieve state-of-the-art predictive accuracy and robustness.

> **Key Innovation:** Unlike traditional methods that struggle with non-stationary disturbances, MCAF introduces **DecompMAE** (a Time-Frequency Masked Autoencoder) and a **Complex Fusion** mechanism to seamlessly integrate multi-domain features.

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

