import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model import build_model
from tensorflow.keras.callbacks import EarlyStopping


def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


# Load data
df = pd.read_csv("data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

values = df[['Value']].values

# Scaling
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

# Sequences
TIME_STEPS = 30
X, y = create_sequences(scaled, TIME_STEPS)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = build_model((X_train.shape[1], X_train.shape[2]))
model.summary()

# Training
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_pred = scaler.inverse_transform(train_pred)
y_train_inv = scaler.inverse_transform(y_train)

test_pred = scaler.inverse_transform(test_pred)
y_test_inv = scaler.inverse_transform(y_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred))
mae = mean_absolute_error(y_test_inv, test_pred)

print("RMSE:", rmse)
print("MAE :", mae)

# Plot
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Value'], label="Actual")
plt.plot(df.index[TIME_STEPS:split+TIME_STEPS], train_pred, label="Train")
plt.plot(df.index[split+TIME_STEPS:], test_pred, label="Test")
plt.legend()
plt.title("LSTM with Attention – Time Series Forecasting")
plt.show()
