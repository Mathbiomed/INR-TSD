# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import random, math
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

def _training_callbacks(checkpoint_path):
    """Checkpoint the best (lowest-loss) weights and stop on stalled loss."""
    return [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        EarlyStopping(
            monitor="loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
    ]

def seasonal_hidden_units(n_observations: int) -> int:
    return math.ceil(n_observations / SEASONAL_WIDTH_DIVISOR)

def fit_trend(x_norm, y_norm):
    """Fit the trend model and return it with the best checkpointed weights."""
    model = build_trend_model()
    seed_first_layer_frequencies(
        model,
        "T_dense1",
        max_frequency=TREND_INIT_MAX_FREQUENCY,
    )

    schedule = WarmUpCosineDecay(
        INITIAL_LR, TREND_TARGET_LR, WARMUP_STEPS, DECAY_STEPS
    )
    model.compile(optimizer=Adam(learning_rate=schedule), loss="mae")
    model.fit(
        x_norm, y_norm,
        epochs=TREND_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=_training_callbacks(TREND_CKPT),
        verbose=0,
    )
    model.load_weights(str(TREND_CKPT))
    return model


def fit_seasonality(x_season, residual):
    """Fit the seasonality model over multiple stages and return it.

    Each stage re-compiles with a fresh optimiser so the warmup/cosine schedule
    restarts, while the learned weights carry over between stages.
    """

    n_units = seasonal_hidden_units(len(x_season))
    model = build_seasonality_model(n_units=n_units)
    seed_first_layer_frequencies(
        model,
        "S_dense1",
        max_frequency=SEASONAL_INIT_MAX_FREQUENCY,
    )

    for epochs in SEASONALITY_EPOCHS:
        schedule = WarmUpCosineDecay(
            INITIAL_LR, SEASONALITY_TARGET_LR, WARMUP_STEPS, DECAY_STEPS
        )
        model.compile(optimizer=Adam(learning_rate=schedule), loss="mae")
        model.fit(
            x_season, residual,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            callbacks=_training_callbacks(SEASONALITY_CKPT),
            verbose=0,
        )
    model.load_weights(str(SEASONALITY_CKPT))
    return model


def decompose_series(x, y):
    """Decompose one series into trend and seasonality (original value scale)."""
    x_norm = normalize_inputs(x)
    y_norm = normalize_targets(y)

    # 1) Fit and subtract the trend.
    trend_model = fit_trend(x_norm, y_norm)
    trend_norm = trend_model.predict(x_norm, verbose=0).reshape(-1)
    residual = y_norm - trend_norm

    # 2) Fit the residual seasonality. The input is rescaled so the sine basis
    #    spans the full length of the series.
    x_season = x * (np.pi * len(x)) / x.max()
    season_model = fit_seasonality(x_season, residual)
    season_norm = season_model.predict(x_season, verbose=0).reshape(-1)

    trend = denormalize_trend(trend_norm, y)
    seasonality = denormalize_seasonality(season_norm, y)
    return trend, seasonality