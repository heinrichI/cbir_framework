import numpy as np

from core.quantization.pq_quantizer import PQQuantizer
from core.transformer.items_transformer import ItemsTransformer


def arrays_to_productbincount(arrays: np.ndarray, pq_quantizer: PQQuantizer):
    subspaced_indices_arr = pq_quantizer.predict_subspace_indices(arrays)

    n_subspaces = subspaced_indices_arr.shape[0]
    # bincount = np.bincount(indices, minlength=nclusters)
    product_bincout = np.empty((n_subspaces, pq_quantizer.n_clusters))
    for i in range(n_subspaces):
        product_bincout[i] = np.bincount(subspaced_indices_arr[i], minlength=pq_quantizer.n_clusters)
    final_bincout = product_bincout.ravel()
    # print(final_bincout.shape)
    return final_bincout

class ArraysToProductBinCount(ItemsTransformer):
    def __init__(self, pq_quantizer: PQQuantizer):
        self.pq_quantizer = pq_quantizer
        self.siftsset_to_productbovwbincount = arrays_to_productbincount

    def transform_item(self, item):
        return arrays_to_productbincount(item, self.pq_quantizer)

