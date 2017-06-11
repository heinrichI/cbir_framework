import collections;
import os

from core.data_store import datastore

from core.common import file_utils


class FileSystemDirectoryDataStore(datastore.DataStore):
    def __init__(self, dir_path, recursive=True, ids_are_fullpathes=False):
        self.dir_path = dir_path
        self.recursive = recursive
        self.ids_are_fullpathes = ids_are_fullpathes

    def _imagename_to_bytes(self, img_name: str) -> bytes:
        img_path = os.path.join(self.dir_path, img_name)
        with open(img_path, "rb") as binary_file:
            data = binary_file.read()
            return data

    def get_ids_sorted(self) -> collections.Iterable:
        list_files_relative_pathes = file_utils.list_files_relative_pathes(self.dir_path, self.recursive)
        image_relative_pathes = filter(file_utils.filter_by_image_extensions, list_files_relative_pathes)
        image_relative_pathes_sorted = sorted(image_relative_pathes)
        if self.ids_are_fullpathes:
            fullpathes=[os.path.join(self.dir_path, rel_path) for rel_path in image_relative_pathes_sorted]
            return fullpathes
        else:
            return image_relative_pathes_sorted

    def get_items_sorted_by_ids(self, ids_sorted=None) -> collections.Iterable:
        if not ids_sorted:
            ids_sorted = self.get_ids_sorted()
        imagebytes_arrays_sorted_by_ids = map(self._imagename_to_bytes, ids_sorted)
        return imagebytes_arrays_sorted_by_ids

    def save_items_sorted_by_ids(self, items_sorted_by_ids, ids_sorted=None):
        raise NotImplemented()

    def get_count(self):
        items = self.get_items_sorted_by_ids()
        count_ = sum(1 for _ in items)
        return count_

    def is_stream_data_store(self):
        return True

    def to_json_dict(self):
        json_dict = {}
        json_dict['kwargs'] = self.__dict__
        json_dict['type'] = 'filedirectory'
        return json_dict
