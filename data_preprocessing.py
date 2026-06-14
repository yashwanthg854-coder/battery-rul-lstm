"""
data_preprocessing.py

Loads NASA Li-ion battery aging dataset (.mat files), extracts discharge cycle
features, computes capacity fade, defines the End-of-Useful-Life (EOL) point
at 80% of initial rated capacity, and builds RUL labels + sliding-window
sequences for LSTM training.

NASA Prognostics Data Repository - Li-ion Battery Aging Dataset
Download: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
Files used here: B0005.mat, B0006.mat, B0007.mat, B0018.mat
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


# ----------------------------------------------------------------------
# 1. Load raw .mat files and extract per-cycle discharge features
# ----------------------------------------------------------------------
def load_battery_mat(filepath, battery_id):
    """
    Parses a NASA battery .mat file and returns a DataFrame with one row
    per DISCHARGE cycle, containing summary statistics that are good
    predictors of capacity fade.

    Columns returned:
        battery_id, cycle, capacity, avg_voltage, min_voltage,
        avg_current, avg_temperature, max_temperature, discharge_time
    """
    raw = loadmat(filepath, simplify_cells=True)
    key = battery_id  # top-level struct is usually named like 'B0005'
    cycles = raw[key]["cycle"]

    records = []
    cycle_num = 0
    for c in cycles:
        if c["type"] != "discharge":
            continue
        cycle_num += 1
        data = c["data"]

        voltage = np.asarray(data["Voltage_measured"]).flatten()
        current = np.asarray(data["Current_measured"]).flatten()
        temperature = np.asarray(data["Temperature_measured"]).flatten()
        time = np.asarray(data["Time"]).flatten()
        capacity = data.get("Capacity", np.nan)
        if isinstance(capacity, (list, np.ndarray)):
            capacity = np.asarray(capacity).flatten()[0]

        records.append({
            "battery_id": battery_id,
            "cycle": cycle_num,
            "capacity": capacity,
            "avg_voltage": np.nanmean(voltage),
            "min_voltage": np.nanmin(voltage),
            "avg_current": np.nanmean(current),
            "avg_temperature": np.nanmean(temperature),
            "max_temperature": np.nanmax(temperature),
            "discharge_time": time[-1] - time[0] if len(time) > 1 else np.nan,
        })

    return pd.DataFrame(records)


def build_dataset(data_dir, battery_ids=("B0005", "B0006", "B0007", "B0018")):
    """Loads and concatenates multiple battery .mat files into one DataFrame."""
    frames = []
    for bid in battery_ids:
        path = os.path.join(data_dir, f"{bid}.mat")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        df = load_battery_mat(path, bid)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# 2. RUL labeling: cycles remaining until capacity drops below 80% of
#    the INITIAL (rated) capacity for that battery.
# ----------------------------------------------------------------------
def add_rul_labels(df, eol_fraction=0.80):
    """
    For each battery:
      - rated_capacity = capacity at cycle 1
      - eol_threshold  = eol_fraction * rated_capacity
      - eol_cycle      = first cycle where capacity <= eol_threshold
                         (if never reached, use last observed cycle)
      - RUL(cycle)     = eol_cycle - cycle  (clipped at 0)
    """
    out = []
    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        rated_capacity = g["capacity"].iloc[0]
        threshold = eol_fraction * rated_capacity

        below = g[g["capacity"] <= threshold]
        eol_cycle = below["cycle"].iloc[0] if len(below) > 0 else g["cycle"].iloc[-1]

        g["rated_capacity"] = rated_capacity
        g["eol_cycle"] = eol_cycle
        g["RUL"] = (eol_cycle - g["cycle"]).clip(lower=0)
        out.append(g)

    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------
# 3. Normalize features and build sliding-window sequences for LSTM
# ----------------------------------------------------------------------
FEATURE_COLS = [
    "capacity", "avg_voltage", "min_voltage",
    "avg_current", "avg_temperature", "max_temperature", "discharge_time",
]


def normalize_features(df, feature_cols=FEATURE_COLS, stats=None):
    """
    Min-max normalize feature columns to [0, 1].
    If `stats` is provided (dict of {col: (min, max)}), reuse it (for test set).
    Returns the normalized df and the stats dict (fit on train if stats=None).
    """
    df = df.copy()
    if stats is None:
        stats = {col: (df[col].min(), df[col].max()) for col in feature_cols}

    for col in feature_cols:
        mn, mx = stats[col]
        rng = (mx - mn) if (mx - mn) != 0 else 1.0
        df[col] = (df[col] - mn) / rng

    return df, stats


def create_sequences(df, feature_cols=FEATURE_COLS, window_size=10):
    """
    Builds sliding-window sequences per battery for LSTM input.

    Returns:
        X: np.ndarray, shape (num_samples, window_size, num_features)
        y: np.ndarray, shape (num_samples,)  -- RUL at the END of each window
        meta: DataFrame with battery_id and cycle for each sample (for plotting)
    """
    X, y, meta = [], [], []

    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle").reset_index(drop=True)
        features = g[feature_cols].values
        ruls = g["RUL"].values
        cycles = g["cycle"].values

        for i in range(window_size, len(g)):
            X.append(features[i - window_size:i])
            y.append(ruls[i])
            meta.append({"battery_id": bid, "cycle": cycles[i]})

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    meta = pd.DataFrame(meta)
    return X, y, meta


# ----------------------------------------------------------------------
# 4. Train / test split by battery (avoid leakage between cycles of the
#    same battery across train and test sets)
# ----------------------------------------------------------------------
def train_test_split_by_battery(df, test_batteries):
    train_df = df[~df["battery_id"].isin(test_batteries)].reset_index(drop=True)
    test_df = df[df["battery_id"].isin(test_batteries)].reset_index(drop=True)
    return train_df, test_df


if __name__ == "__main__":
    DATA_DIR = "../data"
    df = build_dataset(DATA_DIR)
    df = add_rul_labels(df, eol_fraction=0.80)

    train_df, test_df = train_test_split_by_battery(df, test_batteries=["B0018"])

    train_df, stats = normalize_features(train_df)
    test_df, _ = normalize_features(test_df, stats=stats)

    X_train, y_train, meta_train = create_sequences(train_df, window_size=10)
    X_test, y_test, meta_test = create_sequences(test_df, window_size=10)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    np.savez("../data/processed_sequences.npz",
             X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    print("Saved processed sequences to ../data/processed_sequences.npz")
