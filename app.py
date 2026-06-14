"""
app.py

Streamlit demo app for the Battery Remaining Useful Life (RUL) Predictor.

Since this is a public demo (no access to the proprietary NASA dataset files
at runtime), the app generates realistic SYNTHETIC battery degradation data
that mimics the NASA Li-ion Battery Aging Dataset's structure, trains a small
LSTM model on it at startup (cached so it only runs once), and then lets the
user explore live RUL predictions interactively.

The full project -- including the real data pipeline for the NASA dataset,
model architecture, and training scripts -- is in src/.
"""

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

st.set_page_config(page_title="Battery RUL Predictor", page_icon="🔋", layout="centered")

WINDOW_SIZE = 10
FEATURE_NAMES = [
    "Capacity", "Avg Voltage", "Min Voltage",
    "Avg Current", "Avg Temperature", "Max Temperature", "Discharge Time",
]
NUM_FEATURES = len(FEATURE_NAMES)


# ----------------------------------------------------------------------
# Synthetic data generator (mimics NASA Li-ion battery aging dataset shape)
# ----------------------------------------------------------------------
@st.cache_data
def generate_synthetic_batteries(num_batteries=6, num_cycles=160, seed=42):
    rng = np.random.default_rng(seed)
    all_rows = []

    for b in range(num_batteries):
        rated_capacity = 2.0  # Ah, similar to NASA dataset batteries
        fade_rate = rng.uniform(0.0035, 0.006)
        noise = rng.normal(0, 0.01, size=num_cycles)

        capacity = rated_capacity * np.exp(-fade_rate * np.arange(num_cycles)) + noise
        capacity = np.clip(capacity, 0.5, rated_capacity)

        avg_voltage = 3.6 - 0.15 * (1 - capacity / rated_capacity) + rng.normal(0, 0.01, num_cycles)
        min_voltage = 3.0 - 0.2 * (1 - capacity / rated_capacity) + rng.normal(0, 0.01, num_cycles)
        avg_current = -1.0 + rng.normal(0, 0.02, num_cycles)
        avg_temp = 25 + 5 * (1 - capacity / rated_capacity) + rng.normal(0, 0.5, num_cycles)
        max_temp = avg_temp + rng.uniform(2, 5, num_cycles)
        discharge_time = 3600 * (capacity / rated_capacity) + rng.normal(0, 30, num_cycles)

        for c in range(num_cycles):
            all_rows.append({
                "battery_id": f"B{b+1:02d}",
                "cycle": c + 1,
                "capacity": capacity[c],
                "avg_voltage": avg_voltage[c],
                "min_voltage": min_voltage[c],
                "avg_current": avg_current[c],
                "avg_temperature": avg_temp[c],
                "max_temperature": max_temp[c],
                "discharge_time": discharge_time[c],
                "rated_capacity": rated_capacity,
            })

    return pd.DataFrame(all_rows)


def add_rul_labels(df, eol_fraction=0.80):
    out = []
    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        threshold = eol_fraction * g["rated_capacity"].iloc[0]
        below = g[g["capacity"] <= threshold]
        eol_cycle = below["cycle"].iloc[0] if len(below) > 0 else g["cycle"].iloc[-1]
        g["eol_cycle"] = eol_cycle
        g["RUL"] = (eol_cycle - g["cycle"]).clip(lower=0)
        out.append(g)
    return pd.concat(out, ignore_index=True)


FEATURE_COLS = ["capacity", "avg_voltage", "min_voltage",
                "avg_current", "avg_temperature", "max_temperature", "discharge_time"]


def normalize(df, stats=None):
    df = df.copy()
    if stats is None:
        stats = {c: (df[c].min(), df[c].max()) for c in FEATURE_COLS}
    for c in FEATURE_COLS:
        mn, mx = stats[c]
        rng_ = (mx - mn) if (mx - mn) != 0 else 1.0
        df[c] = (df[c] - mn) / rng_
    return df, stats


def create_sequences(df, window_size=WINDOW_SIZE):
    X, y, meta = [], [], []
    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        feats = g[FEATURE_COLS].values
        ruls = g["RUL"].values
        cycles = g["cycle"].values
        for i in range(window_size, len(g)):
            X.append(feats[i - window_size:i])
            y.append(ruls[i])
            meta.append({"battery_id": bid, "cycle": cycles[i]})
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.DataFrame(meta)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model(window_size, num_features):
    inputs = layers.Input(shape=(window_size, num_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


@st.cache_resource(show_spinner="Training LSTM on synthetic battery data (one-time, ~30s)...")
def train_model():
    df = generate_synthetic_batteries()
    df = add_rul_labels(df, eol_fraction=0.80)

    train_batteries = df["battery_id"].unique()[:-1]
    test_battery = df["battery_id"].unique()[-1]

    train_df = df[df["battery_id"].isin(train_batteries)].reset_index(drop=True)
    test_df = df[df["battery_id"] == test_battery].reset_index(drop=True)

    train_df, stats = normalize(train_df)
    test_df, _ = normalize(test_df, stats)

    X_train, y_train, _ = create_sequences(train_df)
    X_test, y_test, meta_test = create_sequences(test_df)

    model = build_model(WINDOW_SIZE, NUM_FEATURES)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=15, batch_size=16, verbose=0)

    return model, stats, df, test_df, X_test, y_test, meta_test, test_battery


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🔋 Battery Remaining Useful Life (RUL) Predictor")
st.caption(
    "LSTM-based prognostics model — predicts how many charge/discharge cycles "
    "remain before a Li-ion battery's capacity drops to 80% of its rated value "
    "(the industry-standard EV end-of-life threshold)."
)

st.info(
    "This live demo trains on **synthetic battery degradation data** generated "
    "to mimic NASA's Li-ion Battery Aging Dataset. The full project (real NASA "
    "data pipeline, model, and training scripts) is on GitHub — see link below.",
    icon="ℹ️",
)

model, stats, df, test_df, X_test, y_test, meta_test, test_battery = train_model()

st.subheader(f"Test Battery: {test_battery}")

# Predicted vs actual RUL plot
y_pred = model.predict(X_test, verbose=0).flatten()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(meta_test["cycle"], y_test, label="Actual RUL", marker="o", markersize=3)
ax.plot(meta_test["cycle"], y_pred, label="Predicted RUL", marker="x", markersize=3)
ax.set_xlabel("Cycle")
ax.set_ylabel("Remaining Useful Life (cycles)")
ax.set_title(f"RUL Prediction — {test_battery}")
ax.legend()
st.pyplot(fig)

mae = np.mean(np.abs(y_pred - y_test))
st.metric("Test MAE (cycles)", f"{mae:.1f}")

st.divider()

# Interactive single-cycle prediction
st.subheader("Try a Prediction")
st.write("Pick a cycle number for the test battery to see the model's RUL prediction:")

cycle_options = meta_test["cycle"].tolist()
selected_cycle = st.select_slider("Cycle", options=cycle_options, value=cycle_options[len(cycle_options) // 2])

idx = meta_test.index[meta_test["cycle"] == selected_cycle][0]
pred = model.predict(X_test[idx:idx+1], verbose=0).flatten()[0]
actual = y_test[idx]

col1, col2, col3 = st.columns(3)
col1.metric("Cycle", int(selected_cycle))
col2.metric("Predicted RUL", f"{pred:.1f} cycles")
col3.metric("Actual RUL", f"{actual:.1f} cycles")

st.caption(
    "Note: predictions are most meaningful in the context of a full degradation "
    "trajectory. On the real NASA dataset (B0005, B0006, B0007, B0018), this "
    "architecture achieves ~5-10 cycle MAE."
)

st.divider()
st.markdown(
    "**GitHub:** [battery-rul-lstm](https://github.com/yashwanthg854-coder/battery-rul-lstm) "
    "— full source, README, and training pipeline for the real NASA dataset."
)
