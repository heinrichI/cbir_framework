import numpy as np

from core.transformer.items_transformer import ItemsTransformer


def bytes_to_ndarray(image_bytes: bytes, dtype=np.uint8) -> np.ndarray:
    ndarr = np.fromstring(image_bytes, dtype)
    return ndarr


class BytesToNdarray(ItemsTransformer):
    def __init__(self):
        self.bytes_to_ndarray = bytes_to_ndarray

    def transform_item(self, item):
        return self.bytes_to_ndarray(item)