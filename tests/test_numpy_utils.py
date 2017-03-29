import unittest
import numpy as np
from common.numpy_utils import translate_matrix_by_keys

class TestTranslateByKeys(unittest.TestCase):

    def test_translate_matrix_by_keys(self):
        id_stream = range(10)
        sourceid_stream =np.array([str(x * 2) + ".jpg" for x in range(10)])
        matrix_to_translate = np.array([
            [3, 2, 0],
            [1, 4, 2]
        ]
        )
        translated_array = translate_matrix_by_keys(id_stream, sourceid_stream, matrix_to_translate)
        truth_array = np.array([
            ["6.jpg", "4.jpg", "0.jpg"],
            ["2.jpg", "8.jpg", "4.jpg"]
        ])
        self.assertTrue((translated_array == truth_array).all())


if __name__ == '__main__':
    unittest.main()