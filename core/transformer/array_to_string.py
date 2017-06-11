import numpy as np
import io
import io
from core.transformer.items_transformer import ItemsTransformer


def ndarray_to_string(arr: np.ndarray):
    fmt_params = {}
    if issubclass(arr.dtype.type, np.integer):
        fmt_params['fmt'] = "%i"
    elif issubclass(arr.dtype.type, np.floating):
        fmt_params['fmt'] = "%f"

    s = io.BytesIO()
    np.savetxt(s, arr, **fmt_params)
    arrstr = str(s.getvalue())
    return arrstr


class ArrayToStringTransformer(ItemsTransformer):
    def transform_item(self, item):
        return ndarray_to_string(item)
