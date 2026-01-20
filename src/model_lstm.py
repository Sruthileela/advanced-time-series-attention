import tensorflow as tf
from tensorflow.keras import layers, Model

def build_lstm_model(input_shape, lstm_units=[64, 32], dropout_rate=0.2):
    """Build LSTM model for time series forecasting."""
    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    for i, units in enumerate(lstm_units):
        return_sequences = (i < len(lstm_units) - 1)
        x = layers.LSTM(units, return_sequences=return_sequences,
                       dropout=dropout_rate)(x)
    
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1)(x)  # Single step prediction
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_lstm_attention_model(input_shape, lstm_units=128):
    """Build LSTM with attention mechanism."""
    inputs = layers.Input(shape=input_shape)
    
    # LSTM layer
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(lstm_out)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(lstm_units)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention
    attended = layers.multiply([lstm_out, attention])
    attended = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)
    
    # Output layer
    outputs = layers.Dense(1)(attended)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model
