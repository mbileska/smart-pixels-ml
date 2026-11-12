"""Checks that the paired Keras baseline consumes and reports LNN data correctly."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from .reference_baseline import load_split, minimum_interval_68


class ReferenceBaselineTest(unittest.TestCase):
    def test_channels_last_layout_and_label_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            x = np.zeros((2, 2, 16, 16), dtype=np.uint8)
            x[0, 0, 3, 7] = 1
            x[0, 1, 5, 9] = 3
            y = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
            np.savez_compressed(folder / "shard_00000.npz", x=x, y=y)
            images, labels = load_split(folder)
            self.assertEqual(images.shape, (2, 16, 16, 2))
            self.assertEqual(float(images[0, 3, 7, 0]), 1)
            self.assertEqual(float(images[0, 5, 9, 1]), 3)
            np.testing.assert_array_equal(labels, y)

    def test_minimum_68_interval(self):
        residual = np.array([-10, -1, 0, 1, 2, 10])
        self.assertEqual(minimum_interval_68(residual), (-1.0, 10.0))


if __name__ == "__main__":
    unittest.main()
