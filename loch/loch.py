# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

# If you use this implementation, please cite
# Samuel G. Fadel, Francisco M. Fatore, Felipe S.L.G. Duarte, Fernando V. Paulovich, LoCH: A neighborhood-based
# multidimensional projection technique for high-dimensional sparse spaces, Neurocomputing, Volume 150, Part B, 2015,
# Pages 546-556, https://doi.org/10.1016/j.neucom.2014.07.071.

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.neighbors import KDTree
import math

epsilon = 1e-5


def get_distance_function(metric):
    match metric:
        case 'braycurtis':
            return distance.braycurtis
        case 'canberra':
            return distance.canberra
        case 'chebyshev':
            return distance.chebyshev
        case 'cityblock':
            return distance.cityblock
        case 'correlation':
            return distance.correlation
        case 'cosine':
            return distance.cosine
        case 'euclidean':
            return distance.euclidean
        case 'jaccard':
            return distance.jaccard
        case 'minkowski':
            return distance.minkowski
        case 'sqeuclidean':
            return distance.sqeuclidean
        case _:
            raise ValueError('Distance metric not supported.')


class LoCH:

    def __init__(self,
                 n_iterations=100,
                 n_neighbors=10,
                 n_components=2,
                 seed=7):

        self.n_iterations_ = n_iterations
        self.n_neighbors_ = n_neighbors
        self.n_components_ = n_components
        self.seed_ = seed
        self.embedding_ = None

    def _fit(self, X, X_sample, y_sample, metric):
        if len(X) < self.n_neighbors_:
            raise ValueError('The n_neighbors need to be small than the X.')

        # start with PCA
        self.embedding_ = PCA(n_components=2, svd_solver='full').fit_transform(X)

        return self

    def transform(self, X, X_sample=None, y_sample=None, metric='euclidean'):
        size = len(X)
        distance_function = get_distance_function(metric)

        # create the Laplacian part of A
        tree = KDTree(X, leaf_size=2, metric=metric)
        dists, indexes = tree.query(X, k=self.n_neighbors_ + 1)

        for k in range(self.n_iterations_):
            for i in range(size):
                x0 = 0
                y0 = 0

                for j in range(self.n_neighbors_):
                    x0 = x0 + self.embedding_[indexes[i][j]][0]
                    y0 = y0 + self.embedding_[indexes[i][j]][1]

                x0 = x0 / self.n_neighbors_
                y0 = y0 / self.n_neighbors_

                vx = self.embedding_[i][0] - x0
                vy = self.embedding_[i][1] - y0
                norm = math.sqrt(vx*vx + vy*vy)
                vx = vx / norm
                vy = vy / norm

                disp = 0

                for j in range(self.n_neighbors_):
                    delta = distance_function(X[i], X[indexes[i][j]])
                    xjx = self.embedding_[indexes[i][j]][0]
                    xjy = self.embedding_[indexes[i][j]][1]

                    d1_2 = (xjx - x0) * (xjx - x0) + (xjy - y0) * (xjy - y0)
                    d3 = (xjx - x0) * vx + (xjy - y0) * vy

                    tmp = delta * delta - d1_2 + d3 * d3

                    if tmp > 0:
                        disp = disp + d3 + math.sqrt(tmp)

                self.embedding_[i][0] = x0 + (disp / self.n_neighbors_) / 50 * vx
                self.embedding_[i][1] = y0 + (disp / self.n_neighbors_) / 50 * vy

        return self.embedding_

    def fit_transform(self, X, X_sample=None, y_sample=None, metric='euclidean'):
        self._fit(X, X_sample, y_sample, metric)
        return self.transform(X, metric)

    def fit(self, X, X_sample=None, y_sample=None, metric='euclidean'):
        return self._fit(X, X_sample, y_sample, metric)
