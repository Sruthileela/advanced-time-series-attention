# Advanced Time Series Forecasting with Deep Learning and Explainability

## Project Overview
This project implements an advanced deep learning approach for multivariate time-series forecasting using a Transformer-based neural network. The objective is to accurately predict future values from historical time-series data while maintaining interpretability through explainable AI techniques.

Traditional statistical models such as ARIMA struggle with non-linearity and long-term dependencies. To overcome these limitations, this project uses a Transformer architecture combined with SHAP explainability to analyze feature importance over time.

---

## Key Features
- Multivariate time-series forecasting
- Transformer-based deep learning model
- Sliding window sequence generation
- Robust preprocessing and normalization
- Model evaluation using RMSE and MAE
- Explainability using SHAP (SHapley Additive Explanations)

---

## Model Architecture
- Input embedding layer
- Multi-head self-attention Transformer encoder
- Fully connected output layer
- Optimized using Adam optimizer and MSE loss

---

## Dataset
- Input: `data.csv`
- Multivariate numerical time-series dataset
- Scaled using Min-Max normalization
- Train-test split: 80% training, 20% testing

---

## Evaluation Metrics
The model performance is evaluated using:
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

These metrics provide insight into forecasting accuracy and robustness.

---

## Explainability
To address the black-box nature of deep learning models, SHAP is used to:
- Quantify feature importance
- Explain model predictions over time
- Improve transparency and interpretability

SHAP summary plots visualize how each feature contributes to predictions.

---

## Installation
Install dependencies using:

```bash
pip install -r requirements.txt
