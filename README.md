Advanced Time Series Forecasting with Deep Learning and Explainability

Project Overview

Time series forecasting plays a crucial role in real-world decision-making systems such as energy demand prediction, financial forecasting, healthcare monitoring, and industrial sensor analysis. Traditional statistical models often struggle to model nonlinear relationships and long-term dependencies present in modern datasets.
This project implements an advanced multivariate time series forecasting system using a Transformer-based deep learning architecture combined with Explainable AI (XAI) techniques. The primary objective is to accurately predict future values of a target variable while maintaining transparency and interpretability of model predictions.

Objectives

Forecast future values from historical multivariate time series data

Capture long-term temporal dependencies using attention mechanisms

Compare deep learning performance with a statistical baseline

Provide explainability for model predictions

Build a reproducible and evaluation-ready forecasting pipeline

Dataset Description

The dataset used in this project is a synthetically generated multivariate time series dataset designed to simulate real-world temporal behavior.

Dataset Characteristics:

Over 1500 time-ordered observations

Multiple numerical features

Includes trend, daily seasonality, and weekly seasonality

Gaussian noise added to simulate real-world variability

The dataset generation process is fully reproducible and implemented in generate_data.py. This ensures compliance with dataset size requirements and allows controlled experimentation.

Data Preprocessing

To prepare the dataset for modeling, the following preprocessing steps are applied:

Normalization
All features are scaled using Min-Max normalization to improve training stability and convergence.

Sliding Window Transformation
The time series is converted into supervised learning sequences using a fixed-length sliding window. Each sequence is used to predict the next target value.

Train-Test Split
Data is split chronologically into:

80% training data

20% testing data

This approach prevents data leakage and preserves temporal order.

Model Architecture

The core forecasting model is a Transformer-based neural network, selected for its ability to model long-range dependencies efficiently.

Architecture Components:

Input embedding layer

Transformer encoder with multi-head self-attention

Fully connected output layer

Why Transformer?

Handles long-term dependencies better than RNNs/LSTMs

Avoids van

Hyperparameter tuning was performed using manual grid exploration.
Key parameters such as learning rate, sequence length, number of
attention heads, and model dimension were evaluated.

The final configuration (learning rate = 0.001, sequence length = 24,
model dimension = 64, attention heads = 4) provided the best balance
between convergence speed and validation performance.

This implementation avoids data leakage, ensures reproducibility,
and follows best practices in deep learning-based time series forecasting.
