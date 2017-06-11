import unittest

import numpy as np

from core.search.nearest_search import NearestSearch


class NearestSearchTest(unittest.TestCase):
    def test_findNearestIndices_all(self):
        base_vectors = np.array([
            [0, 2, 4],
            [0, 3, 4],
            [1, 3, 5],
            [1, 3, 6]
        ])
        ns = NearestSearch()

        query_vectors = np.array([
            [0, 2, 4],
            [1, 3, 5]
        ])
        nearest_indices = ns.find_nearest_indices(base_vectors, query_vectors, metric='l1')
        # print(nearest_indices)
        truth_indices = np.array([
            [0, 1, 2, 3],
            [2, 3, 1, 0],
        ])
        self.assertTrue((nearest_indices == truth_indices).all())


if __name__ == '__main__':
    unittest.main()
