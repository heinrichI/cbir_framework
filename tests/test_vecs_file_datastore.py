import unittest
import numpy as np
from core.data_store.vecs_file_stream_datastore import VecsFileStreamDatastore


class VecsFileDatastoreTest(unittest.TestCase):
    fvecs_filepath = r'C:\data\texmex\siftsmall\siftsmall\siftsmall_base.fvecs'
    ivecs_filepath = r'C:\data\texmex\sift\sift\sift_groundtruth.ivecs'
    arr = np.arange(5 * 10, dtype='int32').reshape((5, 10))
    temp_fvecs_filepath = r'temp.ivecs'
    temp_fvecs_n_components = 10
    fvecs_n_components = 128
    ivecs_n_components = 100
    fvecs_n_vectors = 10000
    ivecs_n_vectors = 10000

    @classmethod
    def setUpClass(cls):
        ds = VecsFileStreamDatastore(cls.temp_fvecs_filepath, cls.temp_fvecs_n_components)
        with ds:
            ds.save_items_sorted_by_ids(iter(cls.arr))

    @classmethod
    def tearDownClass(cls):
        import os
        os.remove(cls.temp_fvecs_filepath)

    def test_get_items3(self):
        ds = VecsFileStreamDatastore(self.temp_fvecs_filepath, self.temp_fvecs_n_components)
        with ds:
            self.assertTrue(ds.get_count(), len(self.arr))
            items = ds.get_items_sorted_by_ids()
            for i, item in enumerate(items):
                self.assertTrue(np.array_equal(self.arr[i], item))

    def test_get_items(self):
        ds = VecsFileStreamDatastore(self.fvecs_filepath, self.fvecs_n_components)
        with ds:
            items = ds.get_items_sorted_by_ids()
            items_count = 0
            for item in items:
                self.assertEqual(item.shape, (self.fvecs_n_components,))
                items_count += 1
            self.assertEqual(items_count, self.fvecs_n_vectors)

    def test_get_count2(self):
        ds = VecsFileStreamDatastore(self.ivecs_filepath, self.ivecs_n_components)
        with ds:
            self.assertTrue(ds.get_count(), self.ivecs_n_vectors)

    def test_get_items2(self):
        ds = VecsFileStreamDatastore(self.ivecs_filepath, self.ivecs_n_components)
        with ds:
            items = ds.get_items_sorted_by_ids()
            items_count = 0
            for item in items:
                self.assertEqual(item.shape, (self.ivecs_n_components,))
                items_count += 1
            self.assertEqual(items_count, self.ivecs_n_vectors)


if __name__ == '__main__':
    unittest.main()
