# --------------------------------------------------------------------------- #
# Learning-rate schedule                                                      #
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

class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay toward zero.

    The learning rate rises linearly from ``initial_learning_rate`` to
    ``target_learning_rate`` over ``warmup_steps``, then follows a cosine curve
    down to ~0 by ``decay_steps``.
    """

    def __init__(self, initial_learning_rate, target_learning_rate,
                 warmup_steps, decay_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.target_learning_rate = target_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)

        warmup_lr = (
            self.initial_learning_rate
            + (self.target_learning_rate - self.initial_learning_rate)
            * step / warmup_steps
        )
        decayed_lr = 0.5 * self.target_learning_rate * (
            1 + tf.cos(np.pi * (step - warmup_steps) / (decay_steps - warmup_steps))
        )
        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decayed_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "target_learning_rate": self.target_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
        }