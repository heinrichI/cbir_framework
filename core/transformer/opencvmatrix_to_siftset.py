import cv2
import numpy as np

from core.transformer.items_transformer import ItemsTransformer
from core.transformer.transformer_with_params import TransformerWithParams


def opencvmatrix_to_siftset(matrix: np.ndarray, sifts_extractor) -> np.ndarray:
    kp, descriptors = sifts_extractor.detectAndCompute(matrix, None)
    if descriptors is None:
        # one formal zero sift if no keypoints where found. For example: white blank page
        descriptors = np.zeros((1, 128), dtype=np.float32)

    # print("keypoints count: ", len(kp))
    return descriptors



class OpencvMatrixToSiftsSet(ItemsTransformer, TransformerWithParams):
    def __init__(self, nfeatures=None, nOctaveLayers=None, contrastThreshold=None, edgeThreshold=None, sigma=None,
                 print_info_each_n_items=None):
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma

        # some problems passing None args to SIFT_create
        kwargs = {'nfeatures': nfeatures, 'nOctaveLayers': nOctaveLayers, 'contrastThreshold': contrastThreshold,
                  'edgeThreshold': edgeThreshold, 'sigma': sigma}
        self.kwargs = dict((k, v) for k, v in kwargs.items() if v)
        self.sifts_extractor = cv2.xfeatures2d.SIFT_create(**self.kwargs)
        self.opencvmatrix_to_siftset = opencvmatrix_to_siftset
        self.n_items_transformed = 0

        self.print_info_each_n_items = print_info_each_n_items

    def transform_item(self, item):
        if self.print_info_each_n_items is not None:
            self.n_items_transformed += 1
            if self.n_items_transformed % 100 == 0:
                print("OpencvMatrixToSiftsSet n_items_transformed: ", self.n_items_transformed)
        return self.opencvmatrix_to_siftset(item, self.sifts_extractor)

    def getParamsInfo(self):
        return {
            'type': 'sifts_set',
            'params': self.kwargs
        }
