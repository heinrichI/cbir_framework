import unittest

import numpy as np

from core.quantization.pq_quantizer import PQQuantizer
from core.transformer.arrays_to_productbincount import ArraysToProductBinCount
from core.transformer.arrays_to_productbincount import arrays_to_productbincount


class ArraysToProductBinCountTest(unittest.TestCase):
    X = None
    pq_quantizer = None

    @classmethod
    def setUpClass(cls):
        cls.X = np.array([
            [0, 2, 4, 2],
            [0, 3, 4, 2],
            [1, 3, 5, 1],
            [1, 3, 6, 0]
        ])
        cls.pq_quantizer = PQQuantizer(n_clusters=3, n_quantizers=2)
        cls.pq_quantizer.fit(cls.X)

    def test_arrays_to_product_bin_cont_shape(self):
        product_bincount = arrays_to_productbincount(ArraysToProductBinCountTest.X, ArraysToProductBinCountTest.pq_quantizer)
        self.assertEqual(product_bincount.shape, (6,))

    def test_ArraysToProductBinCount_shape(self):
        transformer1 = ArraysToProductBinCount(ArraysToProductBinCountTest.pq_quantizer)
        product_bincount = transformer1.transform([ArraysToProductBinCountTest.X])
        self.assertEqual(next(product_bincount).shape, (6,))


if __name__ == '__main__':
    unittest.main()
