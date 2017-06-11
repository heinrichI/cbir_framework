from core.transformer.items_transformer import ItemsTransformer
from core.quantization.pq_quantizer import PQQuantizer
import numpy as np
import collections


class ArrayToPQIndices(ItemsTransformer):
    def __init__(self, pq_quantizer: PQQuantizer):
        self.pq_quantizer = pq_quantizer

    def transform_item(self, item: np.ndarray):
        X = item.reshape((1, -1))
        subspaced_indices = self.pq_quantizer.predict_subspace_indices(X)
        subspaced_indices = subspaced_indices.ravel()
        return subspaced_indices

    def transform_chunks_stream(self, chunks_stream: collections.Iterable):
        return map(self.transform_array_of_items, chunks_stream)

    def transform_array_of_items(self, X: np.ndarray):
        subspaced_indices_arr = self.pq_quantizer.predict_subspace_indices_T(X)
        return subspaced_indices_arr
