# --------------------------------------------------------------------------- #
# Environment setup                                                           #
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

def configure_environment():
    """Configure logging, plotting style, determinism flags, and the GPU."""
    os.environ["PYTHONHASHSEED"] = str(GLOBAL_TF_SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    tf.get_logger().setLevel("ERROR")
    plt.style.use("default")

    tf.random.set_seed(GLOBAL_TF_SEED)
    np.random.seed(GLOBAL_TF_SEED)

    _configure_gpu()


def _configure_gpu():
    """Restrict TensorFlow to the first GPU and enable memory growth, if any."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)


def seed_data_pipeline(seed=DATA_SEED):
    """Reset the Python and NumPy RNGs so each dataset is processed identically."""
    random.seed(seed)
    np.random.seed(seed)