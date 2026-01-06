from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention

inputs = Input(shape=(10,1))
lstm_out = LSTM(64, return_sequences=True)(inputs)
attention = Attention()([lstm_out, lstm_out])
output = Dense(1)(attention)

model = Model(inputs, output)
model.compile(optimizer='adam', loss='mse')
