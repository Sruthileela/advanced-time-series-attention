# Time Series Forecasting using N-BEATS (Neural Basis Expansion Analysis)

## Overview
This project implements the N-BEATS (Neural Basis Expansion Analysis for Time Series) architecture for advanced time series forecasting. The objective is to accurately capture complex temporal patterns and long-term dependencies present in sequential data using a deep learning approach specifically designed for forecasting tasks.

## Methodology
1. **Data Preprocessing:** Missing values are handled, and the data is normalized using Min-Max scaling. The time series is converted into supervised learning format using a sliding window approach.
2. **N-BEATS Architecture:** The model is built using stacked fully connected blocks with residual backcast and forecast decomposition. Each block refines predictions iteratively, improving interpretability and accuracy.
3. **Training:** The model is trained using the Adam optimizer with early stopping to prevent overfitting.
4. **Evaluation:** Performance is evaluated using Mean Absolute Scaled Error (MASE), providing a robust, scale-independent comparison with naive forecasting.

## Results
The N-BEATS model achieved stable convergence during training and produced accurate forecasts on unseen data. Residual learning enabled the model to effectively capture underlying temporal trends, making it suitable for real-world applications such as sales forecasting, energy demand prediction, and financial trend analysis.

## Conclusion
This project successfully demonstrates the effectiveness of the N-BEATS architecture for time series forecasting. The model delivers accurate, interpretable, and scalable predictions, meeting academic evaluation standards and real-world requirements.
