import unittest
import numpy as np
from core.evaluation.retrieval_perfomance import calculate_threshold_retrieval_perfomances2, calculate_average_precision2
import unittest

import numpy as np

from core.evaluation.retrieval_perfomance import calculate_threshold_retrieval_perfomances2, \
    calculate_average_precision2


class TestEvaluation(unittest.TestCase):
    def test_calculate_threshold_retrieval_perfomances2(self):
        neighbor_ids = np.array([1, 2, 11, 3, 12, 4, 13, 14, 15, 5])
        positive_ids = [1, 2, 3, 4, 5]
        neutral_ids = np.array([])
        neighbors_count_cutoffs, precisions, recalls = calculate_threshold_retrieval_perfomances2(neighbor_ids,
                                                                                                  positive_ids,
                                                                                                  neutral_ids)
        truth_precisions = np.array([1., 1., 0.66666667, 0.75, 0.6,
                                     0.66666667, 0.57142857, 0.5, 0.44444444, 0.5])
        truth_recalls = np.array([0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 0.8, 0.8, 1.])
        truth_neighbors_count_cutoffs = np.arange(1, len(neighbor_ids) + 1)
        self.assertTrue(np.allclose(truth_precisions, precisions))
        self.assertTrue(np.allclose(truth_recalls, recalls))
        self.assertTrue(np.allclose(truth_neighbors_count_cutoffs, neighbors_count_cutoffs))

    def test_calculate_average_precision2(self):
        neighbor_ids = np.array([1, 2, 11, 3, 12, 4, 13, 14, 15, 5])
        positive_ids = [1, 2, 3, 4, 5]
        neutral_ids = np.array([])
        neighbors_count_cutoffs, precisions, recalls = calculate_threshold_retrieval_perfomances2(neighbor_ids,
                                                                                                  positive_ids,
                                                                                                  neutral_ids)
        average_precisions = calculate_average_precision2(precisions, recalls)
        truth_average_precisions = np.array([0.2, 0.4, 0.4, 0.55, 0.55,
                                             0.68333333, 0.68333333, 0.68333333, 0.68333333, 0.78333333])
        self.assertTrue(np.allclose(truth_average_precisions, average_precisions))


if __name__ == '__main__':
    unittest.main()
