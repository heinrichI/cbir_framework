from core.transformer.items_transformer import ItemsTransformer
import numpy as np
from sklearn import metrics


class VectorsToPairwiseDistances(ItemsTransformer):
    def transform_item(self, item: np.ndarray):
        """
        :param item: vectors of one subspace 
        :return: 
        """
        X = item
        pairwise_distances = metrics.pairwise_distances(X, metric='l2')
        return pairwise_distances
