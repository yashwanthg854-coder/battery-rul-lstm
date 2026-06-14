"""
model.py

Defines the LSTM architecture for Battery Remaining Useful Life (RUL)
regression, plus a helper to compile it.

Input shape:  (batch_size, window_size, num_features)
Output:       single scalar RUL prediction (regression)
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_model(window_size, num_features, lstm_units=(64, 32), dropout=0.2):
    """
    Stacked LSTM regressor.

    Architecture:
        Input -> LSTM(64, return_sequences=True) -> Dropout
              -> LSTM(32) -> Dropout
              -> Dense(16, relu) -> Dense(1, linear)

    Args:
        window_size: number of past cycles in each input sequence
        num_features: number of features per cycle
        lstm_units: tuple of units for stacked LSTM layers
        dropout: dropout rate applied after each LSTM layer

    Returns:
        Uncompiled tf.keras.Model
    """
    inputs = layers.Input(shape=(window_size, num_features), name="cycle_sequence")

    x = layers.LSTM(lstm_units[0], return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(dropout)(x)

    x = layers.LSTM(lstm_units[1], name="lstm_2")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(16, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(1, activation="linear", name="rul_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="battery_rul_lstm")
    return model


def compile_model(model, learning_rate=1e-3):
    """Compiles model with Adam optimizer, MSE loss, and MAE metric."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


if __name__ == "__main__":
    # Quick sanity check with dummy shapes
    WINDOW_SIZE = 10
    NUM_FEATURES = 7

    model = build_lstm_model(WINDOW_SIZE, NUM_FEATURES)
    model = compile_model(model)
    model.summary()
