# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #

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

def sine_activation(x):
    """Sinusoidal activation used to build a Fourier-like basis."""
    return tf.math.sin(x)


def build_trend_model(n_units=TREND_HIDDEN_UNITS):
    """Single-input sine network that regresses the (normalised) trend."""
    inputs = Input(shape=(1,), name="t_input")
    hidden = Dense(
        n_units,
        activation=sine_activation,
        kernel_initializer=initializers.Constant(0.1),
        name="T_dense1",
    )(inputs)
    outputs = Dense(1, activation="linear", name="T_out")(hidden)
    return Model(inputs, outputs, name="Trend")


def build_seasonality_model(n_units):
    """Single-input sine network that regresses the residual seasonality."""
    inputs = Input(shape=(1,), name="t_input")
    hidden = Dense(n_units, activation=sine_activation, name="S_dense1")(inputs)
    # No explicit output intercept; this does not enforce zero sample mean.
    outputs = Dense(1, activation="linear", use_bias=False, name="S_out")(hidden)
    return Model(inputs, outputs, name="Seasonality")

def seed_first_layer_frequencies(model, layer_name, max_frequency):
    """Initialize weights in [0, max_frequency]; frequencies remain trainable."""
    layer = model.get_layer(layer_name)
    weights, bias = layer.get_weights()
    n_units = layer.units

    weights[0, :] = np.linspace(0.0, max_frequency, n_units)
    layer.set_weights([weights, bias])
