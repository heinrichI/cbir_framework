import itertools
import os
import unittest

import numpy as np

from core.data_store.sqlite_table_datastore import SQLiteTableDataStore
from core.data_store.csv_datastore import CSVDataStore


class CSVDataStoreTestCase(unittest.TestCase):
    file_path = "temp_csv.csv"
    items = np.arange(10 * 5).reshape((10, 5))

    @classmethod
    def setUpClass(cls):
        ds = CSVDataStore(cls.file_path, 'ndarray', 'int32')
        with ds:
            ds.save_items_sorted_by_ids(cls.items)
        # pass

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.file_path)
        pass

    def test_setUpClass_tearDownClass(self):
        pass

    def test_get_count(self):
        # pass
        ds = CSVDataStore(self.file_path, 'ndarray', 'int32')
        with ds:
            count_ = ds.get_count()
            self.assertEqual(count_, len(self.items))

    def test_get_items(self):
        # pass
        ds = CSVDataStore(self.file_path, 'ndarray', 'int32')
        with ds:
            items = ds.get_items_sorted_by_ids()
            for item1, item2 in zip(items, self.items):
                self.assertTrue(np.array_equal(item1, item2))


if __name__ == '__main__':
    unittest.main()
