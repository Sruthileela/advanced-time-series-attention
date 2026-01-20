"""
Generate synthetic time series data for testing and demonstration.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_timeseries(n_samples=10000, n_features=10, 
                                 freq='H', noise_level=0.1):
    """
    Generate multivariate synthetic time series with:
    - Trend components
    - Seasonality (daily, weekly)
    - Noise
    - Correlations between features
    """
    
    # Generate time index
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_samples, freq=freq)
    
    # Base signals
    time = np.arange(n_samples)
    
    # Create correlated features
    np.random.seed(42)
    base_signal = (
        10 * np.sin(2 * np.pi * time / 24) +  # Daily seasonality
        5 * np.sin(2 * np.pi * time / (24 * 7)) +  # Weekly seasonality
        0.01 * time +  # Trend
        np.random.normal(0, noise_level, n_samples)  # Noise
    )
    
    # Generate correlated features
    data = {}
    for i in range(n_features):
        if i == 0:
            # First feature is the base signal
            data[f'feature_{i}'] = base_signal
        else:
            # Other features are correlated with base + unique patterns
            correlation = 0.7 + 0.3 * np.random.random()
            unique_pattern = np.sin(2 * np.pi * time / (24 * (i+1)))
            data[f'feature_{i}'] = (
                correlation * base_signal +
                (1 - correlation) * unique_pattern +
                np.random.normal(0, noise_level * 0.5, n_samples)
            )
    
    # Add target variable (function of multiple features + noise)
    target_weights = np.random.random(n_features)
    target_weights = target_weights / target_weights.sum()
    
    target = np.zeros(n_samples)
    for i in range(n_features):
        target += target_weights[i] * data[f'feature_{i}']
    
    # Add some non-linearity
    target = target + 0.1 * target**2
    
    # Add target to data
    data['target'] = target
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Add date features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Introduce missing values (5% missing)
    mask = np.random.random(df.shape) < 0.05
    df[mask] = np.nan
    
    return df

def save_synthetic_data(filepath='data/synthetic/synthetic_timeseries.csv'):
    """Generate and save synthetic data."""
    print("Generating synthetic time series data...")
    df = generate_synthetic_timeseries(
        n_samples=8760,  # 1 year of hourly data
        n_features=12,
        freq='H',
        noise_level=0.5
    )
    
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to CSV
    df.to_csv(filepath)
    print(f"Synthetic data saved to {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    return df

if __name__ == '__main__':
    # Generate and save data
    df = save_synthetic_data()
    
    # Display sample
    print("\nSample of generated data:")
    print(df.head())
    
    # Display statistics
    print("\nData statistics:")
    print(df.describe())
