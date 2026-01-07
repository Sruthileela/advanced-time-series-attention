
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, units, theta_dim, backcast_length, forecast_length):
        super().__init__()
        self.fc1 = Dense(units, activation="relu")
        self.fc2 = Dense(units, activation="relu")
        self.theta = Dense(theta_dim)

        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        theta = self.theta(x)

        backcast = theta[:, :self.backcast_length]
        forecast = theta[:, self.backcast_length:]

        return backcast, forecast


def build_nbeats(input_size, units=256, stacks=3):
    forecast_length = 1
    backcast_length = input_size
    theta_dim = backcast_length + forecast_length

    inputs = tf.keras.Input(shape=(input_size,))
    residual = inputs
    forecast = tf.zeros_like(inputs[:, :forecast_length])

    for _ in range(stacks):
        block = NBeatsBlock(units, theta_dim, backcast_length, forecast_length)
        backcast, block_forecast = block(residual)
        residual = residual - backcast
        forecast = forecast + block_forecast

    model = Model(inputs, forecast)
    model.compile(optimizer="adam", loss="mse")
    return model
