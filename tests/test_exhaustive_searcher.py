import unittest

import numpy as np

from core.search import exhaustive_searcher


class ExhaustiveSearcherTest(unittest.TestCase):
    def test_exhaustive_searcher(self):
        base_vectors = np.array([
            [0, 2, 4, 5],
            [0, 5, 4, 10],
            [10, 2, 60, 5],
            [10, 5, 60, 10]
        ])
        ids_ndarray = np.arange(1, 5)

        query_vectors = np.array([
            [0, 2, 4, 5],
            [10, 2, 60, 5]
        ])

        searcher = exhaustive_searcher.ExhaustiveSearcher(items_ndarray=base_vectors, ids_ndarray=ids_ndarray)
        nearest_ids_ndarray = searcher.find_nearest_ids(query_vectors, metric='l1')
        truth_indices = np.array([
            [0, 1, 2, 3],
            [2, 3, 0, 1],
        ])
        truth_ids = truth_indices + 1
        self.assertTrue((truth_ids == nearest_ids_ndarray).all())


if __name__ == '__main__':
    unittest.main()
