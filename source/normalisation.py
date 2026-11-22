# --------------------------------------------------------------------------- #
# Normalisation helpers                                                       #
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

def normalize_inputs(x):
    """Scale the time axis to [0, 1]."""
    return (x - x.min()) / (x.max() - x.min())


def normalize_targets(y):
    """Scale the target values to [-1, 1]."""
    return 2 * (y - y.min()) / (y.max() - y.min()) - 1


def denormalize_trend(trend_norm, y):
    """Map a trend learned in [-1, 1] back to the original value range."""
    return ((trend_norm + 1) / 2) * (y.max() - y.min()) + y.min()


def denormalize_seasonality(season_norm, y):
    """Map a zero-centred seasonal component back to the original value range."""
    return (season_norm / 2) * (y.max() - y.min())