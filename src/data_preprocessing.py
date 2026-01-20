import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

class TimeSeriesScaler:
    """Custom scaler for time series data."""
    
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
    
    def fit(self, X, y):
        self.feature_scaler.fit(X)
        self.target_scaler.fit(y.reshape(-1, 1))
        return self
    
    def transform(self, X, y):
        X_scaled = self.feature_scaler.transform(X)
        y_scaled = self.target_scaler.transform(y.reshape(-1, 1))
        return X_scaled, y_scaled.flatten()
    
    def inverse_transform_y(self, y):
        return self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()

def load_and_preprocess(filepath):
    """Load and preprocess time series data."""
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # Handle missing values
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df.select_dtypes(include=[np.number])),
        columns=df.select_dtypes(include=[np.number]).columns
    )
    
    # Add date features
    df_imputed['hour'] = df['date'].dt.hour
    df_imputed['day_of_week'] = df['date'].dt.dayofweek
    df_imputed['month'] = df['date'].dt.month
    
    return df_imputed

def create_sequences(data, lookback=24, forecast_horizon=1, target_col='PM2.5'):
    """Create sequences for time series forecasting."""
    sequences = []
    targets = []
    
    for i in range(len(data) - lookback - forecast_horizon):
        seq = data.iloc[i:i+lookback].values
        target = data.iloc[i+lookback:i+lookback+forecast_horizon][target_col].values
        
        sequences.append(seq)
        targets.append(target)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    # Scale data
    scaler = TimeSeriesScaler()
    X_flat = X.reshape(-1, X.shape[-1])
    X_scaled_flat, _ = scaler.fit_transform(X_flat, y.flatten())
    X_scaled = X_scaled_flat.reshape(X.shape)
    
    return X_scaled, y, scaler
