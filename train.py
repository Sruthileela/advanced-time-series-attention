import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from model import build_nbeats


def create_supervised(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


def mase(y_true, y_pred, y_train):
    naive_forecast = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    return np.mean(np.abs(y_true - y_pred)) / naive_forecast


# Load dataset
df = pd.read_csv("data.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

values = df["Value"].values.reshape(-1, 1)

# Scaling
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# Prepare data
WINDOW = 30
X, y = create_supervised(scaled, WINDOW)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = X_train.reshape(X_train.shape[0], WINDOW)
X_test = X_test.reshape(X_test.shape[0], WINDOW)

# Model
model = build_nbeats(WINDOW)
model.summary()

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Predict
y_pred = model.predict(X_test)

y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Metrics
mase_value = mase(y_test_inv, y_pred_inv, y_train)
print("MASE:", mase_value)

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Forecast")
plt.legend()
plt.title("N-BEATS Forecast")
plt.show()
