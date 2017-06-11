import numpy as np
import io
from core.transformer.items_transformer import ItemsTransformer


def string_to_array(arrstr, dtype=None):
    s = io.BytesIO(eval(arrstr))
    arr_restored = np.loadtxt(s, dtype=dtype)
    return arr_restored


class StringToArrayTransformer(ItemsTransformer):
    def __init__(self, dtype):
        self.dtype = dtype

    def transform_item(self, item):
        return string_to_array(item, self.dtype)
