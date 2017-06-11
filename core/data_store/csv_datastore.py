from core.data_store.datastore import DataStore
from itertools import count
import csv
from core.transformer.array_to_string import ArrayToStringTransformer
from core.transformer.builtin_to_string import BuiltinToString
from core.transformer.string_to_array import StringToArrayTransformer
from core.transformer.string_to_builtin import StringToBuiltin
from core.common.itertools_utils import front
import numpy as np
import os.path
from core.common.file_utils import make_if_not_exists

class CSVDataStore(DataStore):
    def __init__(self, file_path, item_type_read='ndarray', ndarray_elem_type_read='float32', add_extension=True):
        self.file_path = file_path
        self.fieldnames = ['id', 'item']
        self.item_type_read = item_type_read
        self.ndarray_elem_type_read = ndarray_elem_type_read
        self.need_reload = True
        if add_extension and '.' not in self.file_path:
            self.file_path += '.csv'

    def __enter__(self):
        make_if_not_exists(self.file_path)
        self.csvfile = open(self.file_path, 'a+')
        return self

    def __exit__(self, type, value, traceback):
        self.csvfile.close()

    def load_lists_if_need(self):
        if self.need_reload:
            # with open(self.file_path, 'r') as csvfile:
            csvfile = self.csvfile
            csvfile.seek(0)
            reader = csv.reader(csvfile)
            headers = next(reader)
            if self.ndarray_elem_type_read is not None:
                transformer = StringToArrayTransformer(self.ndarray_elem_type_read)
            else:
                transformer = StringToBuiltin(self.item_type_read)

            self.id_item_dict = {}
            for row in reader:
                item = transformer.transform_item(row[1])
                self.id_item_dict[int(row[0])] = item

            self.need_reload = False

    def save_items_sorted_by_ids(self, items_sorted_by_ids, ids_sorted=None):
        self.need_reload = True
        self.csvfile.seek(0)
        self.csvfile.truncate()
        writer = csv.writer(self.csvfile, lineterminator='\n')

        first_elem, items_sorted_by_ids = front(items_sorted_by_ids)
        if isinstance(first_elem, np.ndarray):
            # item_type_str = str(first_elem.dtype)
            transformer = ArrayToStringTransformer()
        else:
            # item_type_str = str(type(first_elem).__name__)
            transformer = BuiltinToString()

        # writer.writerow(item_type_str)
        # writer.writeheader()
        writer.writerow(self.fieldnames)

        stritems = map(transformer.transform_item, items_sorted_by_ids)
        # for s in stritems:
        #     print(s)

        if ids_sorted is None:
            ids_sorted = count(1)

        id_stritem_stream = zip(ids_sorted, stritems)
        writer.writerows(id_stritem_stream)
        self.csvfile.flush()

    def get_count(self):
        self.load_lists_if_need()
        count_ = len(self.id_item_dict)
        return count_

    def is_stream_data_store(self):
        return False

    def get_ids_sorted(self):
        self.load_lists_if_need()
        return self.id_item_dict.keys()

    def get_items_sorted_by_ids(self, ids_sorted=None):
        self.load_lists_if_need()

        if ids_sorted is not None:
            return iter([self.id_item_dict[id_] for id_ in ids_sorted])
        else:
            return iter([self.id_item_dict[id_] for id_ in sorted(self.id_item_dict)])

    def already_exists(self):
        exists = os.path.isfile(self.file_path)
        return exists

    def is_stream_data_store(self):
        return True


"""
class CSVDataStore(DataStore):
    def __init__(self, file_path, item_type_read=None, ndarray_elem_type_read=None):
        self.file_path = file_path
        self.fieldnames = ['id', 'item']
        self.item_type_read = item_type_read
        self.ndarray_elem_type_read = ndarray_elem_type_read

    def __enter__(self):
        self.csvfile = open(self.file_path, 'a+')
        return self

    def __exit__(self, type, value, traceback):
        self.csvfile.close()

    def save_items_sorted_by_ids(self, items_sorted_by_ids, ids_sorted=None):
        self.csvfile.seek(0)
        self.csvfile.truncate()
        writer = csv.writer(self.csvfile, lineterminator='\n')

        first_elem, items_sorted_by_ids = front(items_sorted_by_ids)
        if isinstance(first_elem, np.ndarray):
            # item_type_str = str(first_elem.dtype)
            transformer = ArrayToStringTransformer()
        else:
            # item_type_str = str(type(first_elem).__name__)
            transformer = BuiltinToString()

        # writer.writerow(item_type_str)
        # writer.writeheader()
        writer.writerow(self.fieldnames)

        stritems = map(transformer.transform_item, items_sorted_by_ids)
        # for s in stritems:
        #     print(s)

        if ids_sorted is None:
            ids_sorted = count(1)

        id_stritem_stream = zip(ids_sorted, stritems)
        writer.writerows(id_stritem_stream)

    def get_count(self):
        self.csvfile.seek(0)
        reader = csv.reader(self.csvfile)
        headers = next(reader)
        count_ = sum(1 for _ in reader)
        return count_

    def is_stream_data_store(self):
        return False

    def get_ids_sorted(self):
        self.csvfile.seek(0)
        reader = csv.reader(self.csvfile)
        headers = next(reader)

        ids = map(lambda row: row[0], reader)
        return ids

    def get_items_sorted_by_ids(self, ids_sorted=None):
        self.csvfile.seek(0)
        reader = csv.reader(self.csvfile)
        headers = next(reader)
        if self.ndarray_elem_type_read is not None:
            transformer = StringToArrayTransformer(self.ndarray_elem_type_read)
        else:
            transformer = StringToBuiltin(self.item_type_read)

        rows = reader
        if ids_sorted is not None:
            idstrs_sorted_list = list(ids_sorted)
            ids_sorted_list = list(map(int, idstrs_sorted_list))
            rows = [row for row in reader if int(row[0]) in ids_sorted_list]
        stritems = map(lambda row: row[1], rows)
        items = map(transformer.transform_item, stritems)
        return items
"""
