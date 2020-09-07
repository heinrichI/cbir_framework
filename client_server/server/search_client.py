import os

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

from client_server.json_socket.jsocket_base import JsonClient
from core.data_store.file_system_directory_datastore import FileSystemDirectoryDataStore


"""
obj = {
    'query_items_ds': {
        'type': 'FileSystemDirectoryDataStore',
        'params': {
            'dir_path': r'C:\data\images\brodatz\data.brodatz\size_213x213'
        }
    },
    'n_nearest': 10
}
"""

if __name__ == '__main__':
    client = JsonClient()
    client.connect()

    # client.send_obj(obj)
    img_dir = r'C:\data\images\brodatz\data.brodatz\size_213x213'
    query_images_filepathes = FileSystemDirectoryDataStore(img_dir, ids_are_fullpathes=True).get_ids_sorted()
    query_msg = {
        'query_image_filepathes': query_images_filepathes,
        'n_nearest': 10
    }
    client.send_obj(query_msg)

    nearest_neighbor_native_ids_list = client.read_obj()
    print(len(nearest_neighbor_native_ids_list))
    print(nearest_neighbor_native_ids_list)

    client.send_obj(True)
    client.close()
