"""Focused checks; TensorFlow installation is required (no silent skips)."""

import unittest

import numpy as np
import tensorflow as tf

from config import TREND_HIDDEN_UNITS, TREND_INIT_MAX_FREQUENCY
from source.models import (
    build_trend_model, build_seasonality_model, seed_first_layer_frequencies,
)
from source.train import seasonal_hidden_units


class ModelTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(2)

    def test_seasonal_width_and_all_frequencies(self):
        for n, expected in [(50, 20), (100, 40), (101, 41), (240, 96), (250, 100)]:
            with self.subTest(observations=n):
                units = seasonal_hidden_units(n)
                self.assertEqual(units, expected)
                model = build_seasonality_model(units)
                seed_first_layer_frequencies(model, "S_dense1", 1.0)
                weights, _ = model.get_layer("S_dense1").get_weights()
                self.assertEqual(weights.shape, (1, expected))
                np.testing.assert_allclose(weights[0], np.linspace(0, 1, expected), atol=1e-7)

    def test_trend_initialization_and_trainability(self):
        for upper in (TREND_INIT_MAX_FREQUENCY, 2 * np.pi, 10.0):
            with self.subTest(upper=upper):
                model = build_trend_model()
                layer = model.get_layer("T_dense1")
                seed_first_layer_frequencies(model, "T_dense1", upper)
                weights, _ = layer.get_weights()
                self.assertEqual(layer.units, TREND_HIDDEN_UNITS)
                self.assertTrue(layer.trainable)
                np.testing.assert_allclose(
                    weights[0], np.linspace(0, upper, TREND_HIDDEN_UNITS), atol=1e-6,
                )

    def test_one_optimizer_update(self):
        x = np.linspace(0, 1, 16, dtype=np.float32).reshape(-1, 1)
        y = np.sin(x)
        model = build_seasonality_model(7)
        seed_first_layer_frequencies(model, "S_dense1", 1.0)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
        loss = model.train_on_batch(x, y)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(tuple(model(x).shape), (16, 1))


if __name__ == "__main__":
    unittest.main()
