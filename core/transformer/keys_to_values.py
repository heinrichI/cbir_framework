import numpy as np
from core.transformer.items_transformer import ItemsTransformer

from core.common import numpy_utils as npu


class TranslateByKeysTransformer(ItemsTransformer):
    def __init__(self, keys: np.ndarray, values: np.ndarray, return_list_of_lists=False):
        self.keys = keys
        self.values = values
        self.return_list_of_lists = return_list_of_lists

    def transform(self, items: np.ndarray):
        return npu.translate_matrix_by_keys(self.keys, self.values, items, self.return_list_of_lists)
