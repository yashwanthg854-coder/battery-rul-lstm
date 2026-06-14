"""
predict.py

Load a trained Battery RUL LSTM model and run inference on new battery
cycle data.

Usage:
    python predict.py --battery B0018 --cycle 100
"""

import os
import json
import pickle
import argparse
import numpy as np
import tensorflow as tf

from data_preprocessing import (
    build_dataset, add_rul_labels, normalize_features, create_sequences,
)

MODEL_DIR = "../models"
DATA_DIR = "../data"


def load_artifacts():
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "battery_rul_lstm.keras"))

    with open(os.path.join(MODEL_DIR, "norm_stats.pkl"), "rb") as f:
        stats = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    return model, stats, config


def predict_for_battery(battery_id, target_cycle=None):
    model, stats, config = load_artifacts()
    window_size = config["window_size"]
    feature_cols = config["feature_cols"]

    df = build_dataset(DATA_DIR, battery_ids=[battery_id])
    df = add_rul_labels(df, eol_fraction=0.80)
    df, _ = normalize_features(df, feature_cols, stats=stats)

    X, y, meta = create_sequences(df, feature_cols, window_size)

    if len(X) == 0:
        raise ValueError(f"Not enough cycles for battery {battery_id} "
                          f"(need > {window_size} discharge cycles)")

    if target_cycle is None:
        idx = -1  # most recent
    else:
        matches = meta.index[meta["cycle"] == target_cycle].tolist()
        if not matches:
            raise ValueError(f"Cycle {target_cycle} not found for battery {battery_id}. "
                              f"Available cycles: {meta['cycle'].min()}-{meta['cycle'].max()}")
        idx = matches[0]

    pred_rul = model.predict(X[idx:idx+1], verbose=0).flatten()[0]
    actual_rul = y[idx]
    cycle = meta["cycle"].iloc[idx]

    return {
        "battery_id": battery_id,
        "cycle": int(cycle),
        "predicted_rul": float(pred_rul),
        "actual_rul": float(actual_rul),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery", required=True, help="Battery ID, e.g. B0018")
    parser.add_argument("--cycle", type=int, default=None,
                         help="Cycle number to predict at (default: most recent)")
    args = parser.parse_args()

    result = predict_for_battery(args.battery, args.cycle)

    print(f"\nBattery: {result['battery_id']}  |  Cycle: {result['cycle']}")
    print(f"Predicted RUL: {result['predicted_rul']:.1f} cycles")
    print(f"Actual RUL:    {result['actual_rul']:.1f} cycles")
    print(f"Error:         {abs(result['predicted_rul'] - result['actual_rul']):.1f} cycles\n")
