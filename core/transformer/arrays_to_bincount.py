import numpy as np

from core.quantization.pq_quantizer import PQQuantizer
from core.transformer.items_transformer import ItemsTransformer


def arrays_to_bincount(arrays: np.ndarray, pq_quantizer: PQQuantizer):
    indices = pq_quantizer.predict(arrays)
    bincount = np.bincount(indices, minlength=pq_quantizer.max_scalar_index + 1)
    return bincount


class ArraysToBinCount(ItemsTransformer):
    def __init__(self, pq_quantizer: PQQuantizer):
        self.pq_quantizer = pq_quantizer
        self.arrays_to_bincount = arrays_to_bincount

    def transform_item(self, item):
        return arrays_to_bincount(item, self.pq_quantizer)
