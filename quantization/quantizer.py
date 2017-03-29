import numpy as np


class Quantizer(object):
    def fit(self, X: np.ndarray) -> None:
        pass

    def get_cluster_centers(self) -> np.ndarray:
        pass

    def predict(self, X) -> np.ndarray:
        pass
