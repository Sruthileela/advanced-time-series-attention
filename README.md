from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention, Flatten, Dropout

# Input shape: 10 timesteps, 1 feature
inputs = Input(shape=(10, 1))

# LSTM layer with dropout
lstm_out = LSTM(64, return_sequences=True)(inputs)
lstm_out = Dropout(0.2)(lstm_out)

# Attention layer
attention = Attention()([lstm_out, lstm_out])
attention_flat = Flatten()(attention)

# Output layer
output = Dense(1, activation='linear')(attention_flat)

# Build and compile model
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Print summary
model.summary()
