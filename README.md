# Advanced Time Series Forecasting with Deep Learning and Explainability

## Project Overview
This project implements an advanced multivariate time-series forecasting system using a Transformer-based deep learning model. The objective is to accurately forecast future values while providing interpretability through SHAP explainability.

## Model Architecture
- Transformer Encoder
- Multi-head Self Attention
- Sliding window sequence modeling

## Dataset
- Multivariate time-series dataset
- Normalized using MinMaxScaler
- 80% training, 20% testing split

## Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

## Explainability
SHAP (SHapley Additive Explanations) is used to interpret feature contributions to model predictions over time.

## How to Run
```bash
pip install -r requirements.txt
python train.py
