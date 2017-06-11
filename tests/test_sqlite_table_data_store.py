import itertools
import os
import unittest

import numpy as np

from core.data_store.sqlite_table_datastore import SQLiteTableDataStore


class SQLiteTableDataStoreNdarrayTestCase(unittest.TestCase):
    """
       create temp database file and table in it with column type: ndarray
    """
    db_path = "temp_sqlite_db"
    table_name = "temp_table"
    items = np.arange(10 * 5).reshape((10, 5))

    @classmethod
    def setUpClass(cls):
        with SQLiteTableDataStore(cls.db_path,
                                  cls.table_name, "ndarray") as sqltable_ds:
            sqltable_ds.save_items_sorted_by_ids(cls.items)

    @classmethod
    def tearDownClass(cls):
        with SQLiteTableDataStore(cls.db_path,
                                  cls.table_name, "ndarray") as sqltable_ds:
            sqltable_ds.drop(True)
        os.remove(cls.db_path+'.sqlite')

    def test_setUpClass_tearDownClass(self):
        pass

    def test_get_items_sorted_by_ids(self):
        with SQLiteTableDataStore(SQLiteTableDataStoreNdarrayTestCase.db_path,
                                  SQLiteTableDataStoreNdarrayTestCase.table_name, "ndarray") as sqltable_ds:
            items_sorted_by_ids = sqltable_ds.get_items_sorted_by_ids()
            for arr1, arr2 in zip(items_sorted_by_ids, SQLiteTableDataStoreNdarrayTestCase.items):
                self.assertTrue((arr1 == arr2).all())

    def test_get_items_sorted_by_ids_particular_ids(self):
        ids_sorted = (x for x in range(1, 11) if x % 2 == 0)
        with SQLiteTableDataStore(SQLiteTableDataStoreNdarrayTestCase.db_path,
                                  SQLiteTableDataStoreNdarrayTestCase.table_name, "ndarray") as sqltable_ds:
            items_sorted_by_ids = sqltable_ds.get_items_sorted_by_ids(ids_sorted)
            items_each_second = itertools.islice(SQLiteTableDataStoreNdarrayTestCase.items, 1, None, 2)
            for arr1, arr2 in zip(items_each_second, items_sorted_by_ids):
                self.assertTrue((arr1 == arr2).all())

    def test_auto_table_names(self):
        with SQLiteTableDataStore(SQLiteTableDataStoreNdarrayTestCase.db_path) as sqltable_ds1,\
                SQLiteTableDataStore(SQLiteTableDataStoreNdarrayTestCase.db_path) as sqltable_ds2:
            sqltable_ds1.save_items_sorted_by_ids(SQLiteTableDataStoreNdarrayTestCase.items)
            sqltable_ds2.save_items_sorted_by_ids(SQLiteTableDataStoreNdarrayTestCase.items)
            items_sorted_by_ids1 = sqltable_ds1.get_items_sorted_by_ids()
            items_sorted_by_ids2 = sqltable_ds2.get_items_sorted_by_ids()
            for arr1, arr2 in zip(items_sorted_by_ids1, items_sorted_by_ids2):
                self.assertTrue((arr1 == arr2).all())


if __name__ == '__main__':
    unittest.main()
