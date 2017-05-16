import unittest

import numpy as np

import common.aggregate_iterable
from common import itertools_utils as iu


class IterToolsTestCase(unittest.TestCase):
    def test_pipeline_empty_transformers(self):
        input_stream = range(6)
        output_stream = iu.pipe_line(input_stream, [])
        self.assertEqual(list(range(6)), list(output_stream))

    def test_pipeline_map_transformer(self):
        input_stream = range(6)
        output_stream = iu.pipe_line(input_stream, [iu.map_transformer(lambda x: x ** 2)])
        self.assertEqual(list(x ** 2 for x in range(6)), list(output_stream))

    def test_aggregate_iterable_int_range_without_count(self):
        x = range(10)
        aggregated = common.aggregate_iterable.aggregate_iterable(x, detect_final_shape_by_first_elem=True)
        truth_val = np.arange(10)
        self.assertTrue((truth_val == aggregated).all())

    def test_aggregate_iterable_int_range_with_count(self):
        x = range(10)
        aggregated = common.aggregate_iterable.aggregate_iterable(x, n_elements=10, detect_final_shape_by_first_elem=True)
        truth_val = np.arange(10)
        self.assertTrue((truth_val == aggregated).all())

    def test_aggregate_iterable_1Darray_stream_without_count(self):
        x = (np.zeros((5,)) for i in range(10))
        aggregated = common.aggregate_iterable.aggregate_iterable(x, detect_final_shape_by_first_elem=True)
        truth_val = np.zeros((10, 5))
        self.assertTrue((truth_val == aggregated).all())

    def test_aggregate_iterable_1Darray_stream_with_count(self):
        x = (np.zeros((5,)) for i in range(10))
        aggregated = common.aggregate_iterable.aggregate_iterable(x, n_elements=10, detect_final_shape_by_first_elem=True)
        truth_val = np.zeros((10, 5))
        self.assertTrue((truth_val == aggregated).all())

    def test_aggregate_iterable_2Darray_stream_without_count(self):
        x = (np.zeros((5, 128)) for i in range(10))
        aggregated = common.aggregate_iterable.aggregate_iterable(x, detect_final_shape_by_first_elem=True)
        truth_val = np.zeros((10, 5, 128))
        self.assertTrue((truth_val == aggregated).all())

    def test_aggregate_iterable_2Darray_stream_with_count(self):
        x = (np.zeros((5, 128)) for i in range(10))
        aggregated = common.aggregate_iterable.aggregate_iterable(x, n_elements=10, detect_final_shape_by_first_elem=True)
        truth_val = np.zeros((10, 5, 128))
        self.assertTrue((truth_val == aggregated).all())


if __name__ == '__main__':
    unittest.main()
