"""
train.py

End-to-end training pipeline for the Battery RUL LSTM model:
  1. Load and preprocess NASA battery data
  2. Build sliding-window sequences
  3. Train LSTM with early stopping
  4. Save trained model + normalization stats
  5. Plot training curves and save to /outputs

Usage:
    python train.py
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from data_preprocessing import (
    build_dataset, add_rul_labels, normalize_features,
    create_sequences, train_test_split_by_battery, FEATURE_COLS,
)
from model import build_lstm_model, compile_model


DATA_DIR = "../data"
MODEL_DIR = "../models"
OUTPUT_DIR = "../outputs"
WINDOW_SIZE = 10
TEST_BATTERIES = ["B0018"]   # held-out battery for testing
EPOCHS = 100
BATCH_SIZE = 16


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load + label
    print("Loading dataset...")
    df = build_dataset(DATA_DIR)
    df = add_rul_labels(df, eol_fraction=0.80)

    # 2. Split by battery (no leakage)
    train_df, test_df = train_test_split_by_battery(df, TEST_BATTERIES)

    # 3. Normalize (fit on train only)
    train_df, stats = normalize_features(train_df, FEATURE_COLS)
    test_df, _ = normalize_features(test_df, FEATURE_COLS, stats=stats)

    # 4. Sliding-window sequences
    X_train, y_train, _ = create_sequences(train_df, FEATURE_COLS, WINDOW_SIZE)
    X_test, y_test, meta_test = create_sequences(test_df, FEATURE_COLS, WINDOW_SIZE)

    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")

    # 5. Build + train model
    model = build_lstm_model(WINDOW_SIZE, num_features=len(FEATURE_COLS))
    model = compile_model(model, learning_rate=1e-3)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    # 6. Evaluate
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {test_loss:.4f} | Test MAE: {test_mae:.4f} cycles")

    # 7. Save model, normalization stats, and metadata
    model.save(os.path.join(MODEL_DIR, "battery_rul_lstm.keras"))

    with open(os.path.join(MODEL_DIR, "norm_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump({
            "window_size": WINDOW_SIZE,
            "feature_cols": FEATURE_COLS,
            "test_batteries": TEST_BATTERIES,
            "test_mae": float(test_mae),
        }, f, indent=2)

    # 8. Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["mae"], label="train")
    axes[1].plot(history.history["val_mae"], label="val")
    axes[1].set_title("MAE (cycles)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    print(f"Saved training curves to {OUTPUT_DIR}/training_curves.png")

    # 9. Plot predicted vs actual RUL on test battery
    y_pred = model.predict(X_test).flatten()

    plt.figure(figsize=(8, 5))
    plt.plot(meta_test["cycle"], y_test, label="Actual RUL", marker="o", markersize=3)
    plt.plot(meta_test["cycle"], y_pred, label="Predicted RUL", marker="x", markersize=3)
    plt.xlabel("Cycle")
    plt.ylabel("Remaining Useful Life (cycles)")
    plt.title(f"RUL Prediction - Battery {TEST_BATTERIES[0]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rul_prediction.png"), dpi=150)
    print(f"Saved RUL prediction plot to {OUTPUT_DIR}/rul_prediction.png")


if __name__ == "__main__":
    main()
