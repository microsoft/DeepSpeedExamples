# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import unittest

import numpy as np

from rollout_prefill import nonnegative_fit


class CalibrationFitTest(unittest.TestCase):

    def test_recovers_nonnegative_costs(self):
        matrix = [[1, 1, 1], [1, 2, 4], [1, 4, 16], [1, 8, 64]]
        np.testing.assert_allclose(nonnegative_fit(matrix, [5.5, 10, 22, 58]), [2, 3, 0.5])
        # Noisy measurements must not produce negative execution costs.
        np.testing.assert_allclose(nonnegative_fit(matrix, [4, 3, 2, 1]), [2.5, 0, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
