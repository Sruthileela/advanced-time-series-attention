import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model import LSTMAttentionModel
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Simulated time series data
np.random.seed(0)
data = np.sin(np.linspace(0, 100, 500)) + 0.1 * np.random.randn(500)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1,1))

# Prepare sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LEN = 20
X, y = create_sequences(data_scaled, SEQ_LEN)

# Convert to torch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Reshape for LSTM (batch, seq_len, features)
X_tensor = X_tensor.view(-1, SEQ_LEN, 1)
y_tensor = y_tensor.view(-1, 1)

# Model
model = LSTMAttentionModel(n_features=1, hidden_dim=32, output_len=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
EPOCHS = 50
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output, attn_weights = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

# Plot predictions
preds, _ = model(X_tensor)
preds = preds.detach().numpy()
plt.plot(y_tensor.detach().numpy(), label='Actual')
plt.plot(preds, label='Predicted')
plt.legend()
plt.show()
