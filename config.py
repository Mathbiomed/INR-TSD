# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

# config.py
import math
from pathlib import Path

# Reproducibility
GLOBAL_TF_SEED = 2
DATA_SEED = 41

# Model - Trend
TREND_HIDDEN_UNITS = 40
# Maximum INITIAL cycles over normalized trend time [0, 1].
# Edit before launching Python; 1.0 means one cycle (omega = 2*pi).
# Default preserves the original frequency range [0, 1].
TREND_INIT_MAX_CYCLES = 1 / (2 * math.pi)
TREND_INIT_MAX_FREQUENCY = 2 * math.pi * TREND_INIT_MAX_CYCLES

# Model - Seasonal
SEASONAL_INIT_MAX_FREQUENCY = 1.0
# Seasonal width: ceil(number of input observations / this divisor).
SEASONAL_WIDTH_DIVISOR = 2.5

# Optimiser schedule
INITIAL_LR = 1e-7
TREND_TARGET_LR = 0.1
SEASONALITY_TARGET_LR = 0.01
WARMUP_STEPS = 100
DECAY_STEPS = 200

# Training
BATCH_SIZE = 500
TREND_EPOCHS = 500
SEASONALITY_EPOCHS = (500, 200, 200)
EARLY_STOPPING_PATIENCE = 1000

# Dataset grid
SEASON_TYPES = (1, 2, 3)
TREND_TYPES = (1, 2, 3)

# Paths
DATASET_DIR = Path("./dataset")
MODEL_DIR = Path("./model")
RESULTS_DIR = Path("./results")
TREND_CKPT = MODEL_DIR / "trend.weights.h5"
SEASONALITY_CKPT = MODEL_DIR / "season.weights.h5"
