import os
import unittest

import numpy as np

from cmd_interface import stepscmd
from core.data_store.sqlite_table_datastore import SQLiteTableDataStore


class StepscmdTest(unittest.TestCase):
    db_path = "temp_sqlite_db.sqlite"
    table_name_in = "temp_table_in"
    table_name_out = "temp_table_out"
    items = np.arange(10 * 8).reshape((10, 8))

    @classmethod
    def setUpClass(cls):
        with SQLiteTableDataStore(StepscmdTest.db_path,
                                  StepscmdTest.table_name_in, "ndarray") as sqltable_ds:
            sqltable_ds.save_items_sorted_by_ids(StepscmdTest.items)

    @classmethod
    def tearDownClass(cls):
        with SQLiteTableDataStore(StepscmdTest.db_path,
                                  StepscmdTest.table_name_in, "ndarray") as sqltable_ds:
            sqltable_ds.drop(True)
        with SQLiteTableDataStore(StepscmdTest.db_path,
                                  StepscmdTest.table_name_out, "ndarray") as sqltable_ds:
            sqltable_ds.drop(True)
        os.remove(StepscmdTest.db_path)

    def test_setUpClass_tearDownClass(self):
        pass

    def test_numpy_transform_step(self):
        items=np.arange(60).reshape(3,20)
        args = stepscmd.parser.parse_args(['transform', '--ds_in', 'numpy', 'np.arange(60).reshape(3,20)',
                                              '--ds_out', 'numpy',
                                              '--trs', 'reshaper', '(5,4)',
                                              '--trs', 'reshaper', '(4,5)',
                                           ])
        # print(results)
        args.func(args)

        transformed_items = args.ds_out.get_items_sorted_by_ids()
        for item, transformed_item in zip(iter(items), transformed_items):
            self.assertEqual(transformed_item.shape, (4, 5))
            self.assertTrue(np.array_equal(transformed_item.ravel(), item.ravel()))
            # steps.transform_step(results.ds_in, results.trs, results.ds_out)

    def test_sql_transform_step(self):
        args = stepscmd.parser.parse_args(['transform', '--ds_in', 'sqlite', StepscmdTest.db_path, StepscmdTest.table_name_in,
                                              '--ds_out', 'sqlite', StepscmdTest.db_path, StepscmdTest.table_name_out,
                                              '--trs', 'reshaper', '(4,2)',
                                              '--trs', 'reshaper', '(2,4)',
                                           ])

        # print(results)
        args.func(args)

        with SQLiteTableDataStore(StepscmdTest.db_path,
                                  StepscmdTest.table_name_out, "ndarray") as sqltable_out_ds:
            transformed_items = sqltable_out_ds.get_items_sorted_by_ids()
            for item, transformed_item in zip(StepscmdTest.items, transformed_items):
                self.assertEqual(transformed_item.shape, (2, 4))
                self.assertTrue(np.array_equal(transformed_item.ravel(), item.ravel()))
                # steps.transform_step(results.ds_in, results.trs, results.ds_out)


if __name__ == '__main__':
    unittest.main()
