import unittest

import numpy as np

from calibration import brier, ece, log_loss, reliability_table


class CalibrationTests(unittest.TestCase):
    def test_hand_example(self):
        # Bin [0.2, 0.3): predictions 0.2, 0.2, outcomes 0, 1 -> gap 0.3, weight 2/4.
        # Bin [0.8, 0.9): predictions 0.8, 0.8, outcomes 1, 1 -> gap 0.2, weight 2/4.
        self.assertAlmostEqual(ece([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1]), 0.25)

    def test_one_goes_in_last_bin(self):
        rows = reliability_table([1.0, 0.95, 0.0], [1, 1, 0], 10)
        self.assertEqual(rows[9][2], 2)
        self.assertEqual(rows[0][2], 1)

    def test_weighting_by_bin_size(self):
        # 99 perfect predictions at 0.05 and one bad one at 0.95: the bad bin weighs 1%.
        p = [0.05] * 99 + [0.95]
        y = [0] * 99 + [0]
        gap_small = abs(0.0 - 0.05)
        expected = 0.99 * gap_small + 0.01 * 0.95
        self.assertAlmostEqual(ece(p, y), expected)

    def test_base_rate_is_calibrated_but_not_sharp(self):
        rng = np.random.default_rng(0)
        q = rng.beta(2, 2, 5000)
        y = (rng.random(5000) < q).astype(float)
        base = np.full(5000, y.mean())
        self.assertAlmostEqual(ece(base, y), 0.0, places=12)
        self.assertLess(brier(q, y), brier(base, y))
        self.assertLess(log_loss(q, y), log_loss(base, y))

    def test_overconfidence_raises_ece(self):
        rng = np.random.default_rng(0)
        q = rng.beta(2, 2, 5000)
        y = (rng.random(5000) < q).astype(float)
        over = np.clip(0.5 + 1.6 * (q - 0.5), 0, 1)
        self.assertAlmostEqual(ece(q, y), 0.013, places=3)
        self.assertAlmostEqual(ece(over, y), 0.090, places=3)

    def test_log_loss_survives_certain_mistakes(self):
        self.assertTrue(np.isfinite(log_loss([1.0, 0.0], [0, 1])))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            ece([1.2], [1])
        with self.assertRaises(ValueError):
            ece([0.5, 0.5], [1])


if __name__ == "__main__":
    unittest.main()
