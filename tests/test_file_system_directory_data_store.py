import shutil
import unittest

from core.data_store.file_system_directory_datastore import *


class TestFileSystemDirectoryDataStore(unittest.TestCase):
    """
    create temp dir and 10 files in it. Write number of file to file
    """
    dir_name = "temp_fsds_test_dir"
    ids_len = 10
    ids_sorted = [str(i) + ".jpg" for i in range(ids_len)]

    @classmethod
    def setUpClass(cls):
        dir_name = cls.dir_name
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for i, id in enumerate(cls.ids_sorted):
            with open(os.path.join(dir_name, id), "wb+") as f:
                f.write(bytearray([i]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TestFileSystemDirectoryDataStore.dir_name)

    def test_setUp_and_teaDown(self):
        pass

    def test_get_ids_sorted(self):
        fsdds = FileSystemDirectoryDataStore(self.__class__.dir_name)
        ids_sorted = fsdds.get_ids_sorted()
        self.assertEqual(TestFileSystemDirectoryDataStore.ids_sorted, ids_sorted)

    def test_get_items_sorted_by_ids(self):
        fsdds = FileSystemDirectoryDataStore(self.__class__.dir_name)
        items_sorted_by_ids = fsdds.get_items_sorted_by_ids()
        for i,item in enumerate(items_sorted_by_ids):
            self.assertEqual(item, bytearray([i]))

    def test_get_items_sorted_by_ids_particular_ids(self):
        fsdds = FileSystemDirectoryDataStore(self.__class__.dir_name)
        ids_sorted = [id for i, id in enumerate(TestFileSystemDirectoryDataStore.ids_sorted) if i % 2 == 0]
        items_sorted_by_ids = fsdds.get_items_sorted_by_ids(ids_sorted)
        for i,item in enumerate(items_sorted_by_ids):
            self.assertEqual(item, bytearray([i*2]))

if __name__ == '__main__':
    unittest.main()
