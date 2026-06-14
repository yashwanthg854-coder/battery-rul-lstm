# Battery Remaining Useful Life (RUL) Prediction using LSTM

Predicts how many charge/discharge cycles remain before a lithium-ion
battery degrades below **80% of its original capacity** — the
industry-standard End-of-Life (EOL) threshold used in EV battery
management systems.

## Overview

| | |
|---|---|
| **Task** | Time-series regression |
| **Model** | Stacked LSTM (64 → 32 units) |
| **Input** | Sliding window of 10 discharge cycles, 7 features per cycle |
| **Output** | RUL in cycles until 80% capacity threshold |
| **Dataset** | [NASA Li-ion Battery Aging Dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (B0005, B0006, B0007, B0018) |

## How It Works

1. **Data extraction** — parses NASA `.mat` files and extracts per-cycle
   summary statistics from each discharge cycle: average/min voltage,
   average current, average/max temperature, discharge time, and measured
   capacity.
2. **RUL labeling** — for each battery, the End-of-Life cycle is defined as
   the first cycle where capacity drops to ≤80% of the cycle-1 (rated)
   capacity. RUL at any cycle = `EOL_cycle - current_cycle`.
3. **Normalization** — min-max scaling fit on training batteries only,
   applied to the held-out test battery to avoid data leakage.
4. **Sequence windows** — 10-cycle sliding windows feed the LSTM, which
   predicts the RUL at the final cycle of each window.
5. **Model** — 2-layer LSTM (64 → 32 units) with dropout, followed by dense
   layers for the final scalar RUL output. Trained with MSE loss and early
   stopping on validation MAE.

## Project Structure

```
battery-rul/
├── data/                   # Place NASA .mat files here (B0005.mat, B0006.mat, ...)
├── models/                 # Saved model, normalization stats, config (generated)
├── outputs/                # Training curves and prediction plots (generated)
├── src/
│   ├── data_preprocessing.py   # Load .mat files, label RUL, build sequences
│   ├── model.py                # LSTM architecture
│   ├── train.py                # Full training pipeline
│   └── predict.py              # Inference on a trained model
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd battery-rul
pip install -r requirements.txt
```

Download the NASA Li-ion battery `.mat` files (B0005, B0006, B0007, B0018)
from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
and place them in `data/`.

## Usage

### Train the model

```bash
cd src
python train.py
```

This will:
- Build train/test splits (B0018 held out as the test battery)
- Train the LSTM with early stopping
- Save the model to `models/battery_rul_lstm.keras`
- Save normalization stats and config to `models/`
- Save training curves and RUL prediction plots to `outputs/`

### Run predictions

```bash
cd src
python predict.py --battery B0018 --cycle 100
```

Output:
```
Battery: B0018  |  Cycle: 100
Predicted RUL: 38.2 cycles
Actual RUL:    41.0 cycles
Error:         2.8 cycles
```

## Results

On the held-out battery (B0018), the model achieves a test **MAE of
~5–10 cycles** depending on random seed and training run — meaning RUL
predictions are typically accurate to within a few charge/discharge cycles.

## Why It's Valuable

- **Predictive maintenance**: enables proactive battery replacement before
  failure, reducing downtime and safety risk.
- **EV fleet management**: helps operators plan battery swaps and resale
  value estimates.
- **Generalizable approach**: the same pipeline (sliding-window LSTM
  regression on degradation features) applies to other equipment
  prognostics problems (motors, turbines, etc.).

## Future Improvements

- Add attention mechanisms or a Transformer-based encoder for longer
  history windows.
- Incorporate impedance/EIS features if available in the dataset.
- Quantify prediction uncertainty (e.g., via quantile regression or
  Monte Carlo dropout) for safety-critical deployment.
