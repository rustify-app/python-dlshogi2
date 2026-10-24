import unittest

import numpy as np

from utils.calibrate_eval_coef import fit_eval_coefficient, log_loss


class EvalCalibrationTest(unittest.TestCase):
    def test_fit_recovers_temperature_from_soft_outcomes(self):
        evaluations = np.linspace(-1600, 1600, 401)
        outcomes = 1.0 / (1.0 + np.exp(-evaluations / 420.0))
        coefficient = fit_eval_coefficient(evaluations, outcomes)
        self.assertAlmostEqual(coefficient, 420.0, delta=1.0)

    def test_calibrated_temperature_improves_log_loss(self):
        evaluations = np.array([-1000, -500, 0, 500, 1000], dtype=float)
        outcomes = np.array([0, 0, 0.5, 1, 1], dtype=float)
        coefficient = fit_eval_coefficient(evaluations, outcomes)
        self.assertLess(log_loss(evaluations, outcomes, coefficient),
                        log_loss(evaluations, outcomes, 600.0))

    def test_rejects_empty_or_malformed_samples(self):
        with self.assertRaisesRegex(ValueError, 'non-empty'):
            fit_eval_coefficient([], [])
        with self.assertRaisesRegex(ValueError, r'in \[0, 1\]'):
            fit_eval_coefficient([1], [2])


if __name__ == '__main__':
    unittest.main()
