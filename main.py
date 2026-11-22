"""Interpretable time-series decomposition with sine-activated neural networks.

Two compact fully-connected networks with sinusoidal activations are fitted to
each series: the first recovers the trend, and the second models the residual
seasonality. The script iterates over a 3x3 grid of (seasonality, trend)
synthetic dataset variants, writes the decomposed components to CSV, and renders
a summary figure.

Expected directory layout (relative to the working directory):
    ./dataset/season{1..3}_trend{1..3}.csv   # inputs: col 0 = time, col 1 = value
    ./model/                                  # checkpoint weights (created if absent)
    ./results/                                # decomposed CSVs and figures (created if absent)

Usage:
    python main.py
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import *
from source.normalisation import *
from source.environment import *
from source.scheduler import *
from source.models import *
from source.train import *

def main():
    configure_environment()
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(18, 18), sharey=True)

    for season_type in SEASON_TYPES:
        for trend_type in TREND_TYPES:
            file_path = DATASET_DIR / f"season{season_type}_trend{trend_type}.csv"

            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"[skip] dataset not found: {file_path} "
                      f"(run data generation first)")
                continue

            seed_data_pipeline(DATA_SEED)

            x = df.iloc[:, 0]
            y = df.iloc[:, 1]

            trend, seasonality = decompose_series(x, y)

            out_path = RESULTS_DIR / f"season{season_type}_trend{trend_type}_decomposed.csv"
            pd.DataFrame({
                "time": x,
                "actual_value": y,
                "predicted_trend": trend,
                "predicted_seasonality": seasonality,
            }).to_csv(out_path, index=False)

            ax = axes[season_type - 1, trend_type - 1]
            ax.plot(x, y, "-o", markersize=4, label="Raw data")
            ax.plot(x, trend, "--", label="Decomposed trend")
            ax.plot(x, seasonality, "--", label="Decomposed seasonality")
            ax.set_title(f"Season {season_type} | Trend {trend_type}", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "decomposed_results.eps", format="eps", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "decomposed_results.png", format="png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
