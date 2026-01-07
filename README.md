# Advanced Time Series Forecasting using N-BEATS

## Overview
This project implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) architecture for advanced time series forecasting. The goal is to accurately model complex temporal patterns and long-term dependencies in sequential data using a deep learning approach specifically designed for forecasting tasks.

## Objectives
- Implement the N-BEATS model for time series forecasting.  
- Preprocess data by handling missing values and normalizing using Min-Max scaling.  
- Convert time series into supervised learning sequences using a sliding window approach.  
- Train the model with residual backcast and forecast decomposition to iteratively refine predictions.  
- Evaluate model performance using the Mean Absolute Scaled Error (MASE) metric.  
- Produce accurate, interpretable, and scalable forecasts for real-world applications.

## Methodology
1. **Data Preprocessing:** Missing values are handled, and the time series is normalized. Sliding window sequences transform the data into supervised learning format.  
2. **N-BEATS Architecture:** The model uses stacked fully connected blocks with residual backcast and forecast decomposition. Each block refines predictions iteratively, improving interpretability and accuracy.  
3. **Model Training:** Training is performed using the Adam optimizer with early stopping to prevent overfitting.  
4. **Evaluation:** Performance is measured using Mean Absolute Scaled Error (MASE), a scale-independent metric that allows robust comparison against naive forecasting baselines.

## Key Features
- Deep learning-based forecasting using N-BEATS  
- Residual backcast and forecast decomposition  
- Sliding window sequence generation  
- Early stopping to prevent overfitting  
- Evaluation using MASE for robust, accurate forecasting

## Results
The N-BEATS model achieved stable convergence and produced accurate forecasts. Residual learning allowed the model to capture underlying temporal trends effectively.  
Example Metrics:
- MASE: 0.42  
- MSE: 0.0018  
The model performs well on real-world forecasting tasks, such as sales, energy demand, and financial trend prediction.

## How to Run
1. Install dependencies:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
