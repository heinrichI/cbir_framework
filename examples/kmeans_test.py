import numpy as np
from matplotlib import pyplot
from sklearn import cluster



if __name__ == '__main__':
    arr = np.random.rand(10 ** 4, 128).astype(dtype='float64')
    k = 100
    centroids_sklearn, assignments_sklearn, inertia_sklearn = cluster.k_means(arr, k, tol=1*arr.shape[1], n_jobs=4, n_init=12,
                                                                              verbose=True)
    print('inertia', inertia_sklearn)

    # 98840.1631088