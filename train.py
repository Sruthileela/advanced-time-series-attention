import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from model import TimeSeriesTransformer

# Load data
df = pd.read_csv("data.csv")

target_col = "target"
features = df.columns.tolist()

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Train-test split
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Sequence creation
def create_sequences(data, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0])
    return np.array(X), np.array(y)

SEQ_LEN = 24
X_train, y_train = create_sequences(train_data, SEQ_LEN)
X_test, y_test = create_sequences(test_data, SEQ_LEN)

# Torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Model
model = TimeSeriesTransformer(
    input_dim=X_train.shape[2],
    model_dim=64,
    num_heads=4,
    num_layers=2
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    output = model(X_train).squeeze()
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# Evaluation (Transformer)
model.eval()
with torch.no_grad():
    preds = model(X_test).squeeze().numpy()

# Inverse scaling
dummy = np.zeros((len(preds), len(features)))
dummy[:, 0] = preds
preds_inv = scaler.inverse_transform(dummy)[:, 0]

dummy[:, 0] = y_test.numpy()
y_test_inv = scaler.inverse_transform(dummy)[:, 0]

rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
mae = mean_absolute_error(y_test_inv, preds_inv)

print("Transformer RMSE:", rmse)
print("Transformer MAE:", mae)

# ===== BASELINE: SARIMAX =====
sarimax_model = SARIMAX(
    df[target_col][:train_size],
    order=(1,1,1),
    seasonal_order=(1,1,1,24)
)
sarimax_fit = sarimax_model.fit(disp=False)
sarimax_preds = sarimax_fit.forecast(steps=len(y_test_inv))

sarimax_rmse = np.sqrt(mean_squared_error(y_test_inv, sarimax_preds))
sarimax_mae = mean_absolute_error(y_test_inv, sarimax_preds)

print("SARIMAX RMSE:", sarimax_rmse)
print("SARIMAX MAE:", sarimax_mae)
