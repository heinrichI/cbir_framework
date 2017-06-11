import unittest
import numpy as np
from core.common.data_wrap import DataWrap


class TestDataWrap(unittest.TestCase):
    def test_chunkify_one_chunk(self):
        it = (np.empty((128,)) for i in range(10))

        dw = DataWrap(it)
        dw.chunkify()
        chunks_stream = dw.data_stream
        for chunk in chunks_stream:
            self.assertEqual(chunk.shape, (10, 128))

    def test_chunkify_two_chunks(self):
        arr = np.random.rand(10, 128)
        it = iter(arr)

        dw = DataWrap(it, items_count=10)
        dw.chunkify(chunk_size=5)
        chunks_stream = dw.data_stream
        for i, chunk in enumerate(chunks_stream):
            self.assertEqual(chunk.shape, (5, 128))
            self.assertTrue(np.array_equal(chunk, arr[i * 5:(i + 1) * 5, :]))

    def test_dechunkify_one_chunk(self):
        arr = np.random.rand(10, 128)
        it = iter(arr)

        dw = DataWrap(it, items_count=-1)
        dw.chunkify(chunk_size=-1)
        dw.dechunkify()
        for i, ai in enumerate(dw.data_stream):
            self.assertTrue(np.array_equal(ai, arr[i]))

    def test_dechunkify_two_chunks(self):
        arr = np.random.rand(10, 128)
        it = iter(arr)

        dw = DataWrap(it, items_count=10)
        dw.chunkify(chunk_size=5)
        dw.dechunkify()
        for i, ai in enumerate(dw.data_stream):
            self.assertTrue(np.array_equal(ai, arr[i]))


if __name__ == '__main__':
    unittest.main()
