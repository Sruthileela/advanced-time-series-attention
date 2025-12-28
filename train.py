import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from model import ProbLSTMAttention

# -----------------------------
# 1. Generate 10 Related Series
# -----------------------------
np.random.seed(42)
t = np.linspace(0, 100, 1000)
data = []

for i in range(10):
    seasonal = np.sin(0.02 * t + i)
    trend = 0.001 * t
    noise = 0.2 * np.random.randn(len(t))
    data.append(seasonal + trend + noise)

df = pd.DataFrame(np.array(data).T, columns=[f"series_{i}" for i in range(10)])

# -----------------------------
# 2. Scale Data
# -----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)

# -----------------------------
# 3. Create sequences
# -----------------------------
SEQ_LEN = 30
X, Y = [], []

for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    Y.append(scaled[i+SEQ_LEN])

X = np.array(X)
Y = np.array(Y)

X_tensor = torch.FloatTensor(X)
Y_tensor = torch.FloatTensor(Y)

dataset = TensorDataset(X_tensor, Y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------------
# 4. Model Setup
# -----------------------------
n_features = df.shape[1]
hidden_dim = 64
output_len = 1
quantiles = [0.1, 0.5, 0.9]

model = ProbLSTMAttention(n_features, hidden_dim, output_len, n_quantiles=len(quantiles))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 5. Quantile Loss
# -----------------------------
def quantile_loss(preds, target, quantiles=[0.1,0.5,0.9]):
    total_loss = 0
    for i, q in enumerate(quantiles):
        e = target - preds[:,:,:,i]
        total_loss += torch.mean(torch.max(q*e, (q-1)*e))
    return total_loss

# -----------------------------
# 6. Training Loop
# -----------------------------
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds, _ = model(xb)
        loss = quantile_loss(preds, yb.unsqueeze(-1), quantiles=quantiles)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(loader):.4f}")

# -----------------------------
# 7. Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    preds, _ = model(X_tensor)
    crps = torch.mean(torch.abs(preds[:,:,:,1] - Y_tensor))
    print("Approx CRPS:", crps.item())

# -----------------------------
# 8. Plot Predictions (Series 0)
# -----------------------------
preds_np = preds.detach().numpy()

plt.figure(figsize=(12,6))
plt.plot(Y[:,0], label="Actual")
plt.plot(preds_np[:,0,0,1], label="Median Prediction")
plt.fill_between(range(len(preds_np)),
                 preds_np[:,0,0,0].flatten(),
                 preds_np[:,0,0,2].flatten(),
                 alpha=0.3, label="P10-P90 Range")
plt.title("Probabilistic Forecast (Series 0)")
plt.legend()
plt.show()
