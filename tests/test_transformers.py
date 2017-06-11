import itertools
import unittest

import numpy as np
from core.transformer.items_transformer import ItemsTransformer

from core.transformer.opencvmatrix_to_glcm import OpencvMatrixToGLCM


def int_to_doubled_int(i):
    return i * 2


class IntToDoubledInt_compose(ItemsTransformer):
    def __init__(self):
        self.int_to_doubled_int_f = int_to_doubled_int

    def transform_item(self, item):
        return self.int_to_doubled_int_f(item)


class TestTransformers(unittest.TestCase):
    def test_doubled(self):
        transformer1 = IntToDoubledInt_compose()
        transformed_items = list(transformer1.transform([1, 2, 3]))
        self.assertEqual([2, 4, 6], transformed_items)

    def test_opencvmatrix_to_glcm(self):
        image = np.array([[0, 0, 1],
                          [0, 0, 1],
                          [2, 2, 3]], dtype=np.uint8)

        images = itertools.repeat(image, 3)
        transformer1 = OpencvMatrixToGLCM(normalize=False, levels=4)
        glcms = transformer1.transform(images)

        truth_results = np.array([2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
        for glcm in glcms:
            self.assertTrue((glcm == truth_results).all())




if __name__ == '__main__':
    unittest.main()
