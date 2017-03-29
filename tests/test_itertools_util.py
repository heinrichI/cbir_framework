import unittest

from common.itertools_utils import *


class IterToolsTestCase(unittest.TestCase):
    def test_pipeline_empty_transformers(self):
        input_stream = range(6)
        output_stream = pipe_line(input_stream, [])
        self.assertEqual(list(range(6)), list(output_stream))

    def test_pipeline_map_transformer(self):
        input_stream = range(6)
        output_stream = pipe_line(input_stream, [map_transformer(lambda x: x ** 2)])
        self.assertEqual(list(x ** 2 for x in range(6)), list(output_stream))

    def test_aggregate_arrays(self):
        array_stream = (np.arange(5) for x in range(10))
        aggregated_array = aggregate_arrays(array_stream)
        self.assertEqual(aggregated_array.shape, (10, 5))

    def test_aggregate_arrays_2dims(self):
        array_stream = (np.arange(10).reshape((2, 5)) for x in range(10))
        aggregated_array = aggregate_arrays(array_stream)
        self.assertEqual(aggregated_array.shape, (10, 2, 5))


if __name__ == '__main__':
    unittest.main()
