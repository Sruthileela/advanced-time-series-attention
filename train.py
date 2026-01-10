import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
SEQ_LEN = 24
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# Dataset
# ----------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ----------------------------
# Transformer Model
# ----------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :]).squeeze()

# ----------------------------
# Load & Preprocess Data
# ----------------------------
df = pd.read_csv("data.csv")

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

train_size = int(0.8 * len(scaled_data))
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

train_ds = TimeSeriesDataset(train_data, SEQ_LEN)
test_ds = TimeSeriesDataset(test_data, SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ----------------------------
# Initialize Model
# ----------------------------
model = TransformerModel(input_dim=df.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    losses = []

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {np.mean(losses):.4f}")

# ----------------------------
# Evaluation
# ----------------------------
model.eval()
preds, actuals = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        output = model(x).cpu().numpy()
        preds.extend(output)
        actuals.extend(y.numpy())

rmse = np.sqrt(mean_squared_error(actuals, preds))
mae = mean_absolute_error(actuals, preds)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# ----------------------------
# Explainability (SHAP)
# ----------------------------
background = torch.tensor(train_data[:100], dtype=torch.float32).to(DEVICE)
test_samples = torch.tensor(test_data[:50], dtype=torch.float32).to(DEVICE)

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_samples)

shap.summary_plot(
    shap_values,
    test_samples.cpu().numpy(),
    feature_names=df.columns
)
