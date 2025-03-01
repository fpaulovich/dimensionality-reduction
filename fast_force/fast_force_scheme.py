# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import numpy as np
import math

from numpy import random
from numba import njit, prange

from scipy.spatial import distance
from sklearn.cluster import BisectingKMeans


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


# @njit(parallel=True, fastmath=False)
def create_distance_matrix(X, metric):
    size = len(X)
    total = int(size * (size + 1) / 2)
    distance_matrix = np.zeros(total, dtype=np.float32)

    distance_function = get_distance_function(metric)

    for i in range(size - 1):
        for j in range(i + 1, size):
            r, s = (i, j) if i < j else (j, i)  # r = min(i,j), s = min(i,j)
            k = int(total - ((size - r) * (size - r + 1) / 2) + (s - r))
            distance_matrix[k] = distance_function(X[i], X[j])

    return distance_matrix


@njit(parallel=True, fastmath=False)
def move2D(ins1, distance_matrix, projection, learning_rate):
    size = len(projection)
    total = len(distance_matrix)
    error = 0

    for ins2 in prange(size):
        if ins1 != ins2:
            x1x2 = projection[ins2][0] - projection[ins1][0]
            y1y2 = projection[ins2][1] - projection[ins1][1]
            dr2 = max(math.sqrt(x1x2 * x1x2 + y1y2 * y1y2), 0.0001)

            # getting the index in the distance matrix and getting the value
            r, s = (ins1, ins2) if ins1 < ins2 else (ins2, ins1)  # r = min(i,j), s = min(i,j)
            d_original = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            # calculate the movement
            delta = (d_original - dr2)
            error += math.fabs(delta)

            # moving
            projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
            projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)

    return error / size


@njit(parallel=True, fastmath=False)
def move(ins1, distance_matrix, projection, learning_rate):
    size = len(projection)
    total = len(distance_matrix)
    error = 0

    for ins2 in prange(size):
        if ins1 != ins2:
            v = projection[ins2] - projection[ins1]
            d_proj = max(np.linalg.norm(v), 0.0001)

            # getting the index in the distance matrix and getting the value
            r, s = (ins1, ins2) if ins1 < ins2 else (ins2, ins1)  # r = min(i,j), s = min(i,j)
            d_original = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            # calculate the movement
            delta = (d_original - d_proj)
            error += math.fabs(delta)

            # moving
            projection[ins2] += learning_rate * delta * (v / d_proj)

    return error / size


# @njit(parallel=False, fastmath=False)
def iteration(distance_matrix, index, projection, learning_rate, n_components):
    error = 0

    for i in range(len(index)):
        ins1 = index[i]

        if n_components == 2:
            error += move2D(ins1, distance_matrix, projection, learning_rate)
        else:
            error += move(ins1, distance_matrix, projection, learning_rate)

    return error / len(index)


class FastForceScheme:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95,
                 tolerance=0.00001,
                 n_iter_without_progress=10,
                 seed=7,
                 n_components=2):

        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.tolerance_ = tolerance
        self.n_iter_without_progress_ = n_iter_without_progress
        self.seed_ = seed
        self.n_components_ = n_components
        self.embedding_ = None

    def _fit(self, X, y, metric):
        size = len(X)

        # create a distance matrix
        distance_matrix = create_distance_matrix(X, metric)

        # set the random seed
        np.random.seed(self.seed_)

        # randomly initialize the projection
        if y is None:
            self.embedding_ = np.random.random((size, self.n_components_))
        else:
            if np.shape(y)[1] == self.n_components_:
                self.embedding_ = y
            else:
                raise ValueError('The n_components should be equal to the number of dimensions of the input embedding.')

        partition_size = int(math.sqrt(size))

        # # calculating index based on k-means
        # index = np.zeros(partition_size).astype(int)
        # bisect_means = BisectingKMeans(n_clusters=partition_size,
        #                                init='random',
        #                                n_init=10,
        #                                bisecting_strategy='largest_cluster',
        #                                random_state=42).fit(X)
        #
        # # getting the point closest to the centroid of each cluster
        # for i in range(partition_size):
        #     index[i] = np.argmin(distance.cdist(bisect_means.cluster_centers_[i].reshape(1, -1), X, metric))

        # iterate until max_it or if the error does not change more than the tolerance
        error = math.inf
        iter_without_progress = 0
        for k in range(self.max_it_):
            index = (np.random.random_sample(partition_size) * size).astype(int)

            learning_rate = self.learning_rate0_ * math.pow((1 - k / self.max_it_), self.decay_)
            new_error = iteration(distance_matrix, index, self.embedding_, learning_rate, self.n_components_)

            if math.fabs(new_error - error) < self.tolerance_:
                iter_without_progress = iter_without_progress + 1
            else:
                iter_without_progress = 0

            if iter_without_progress >= self.n_iter_without_progress_:
                break

            error = new_error

        # setting the min to (0,0)
        self.embedding_ = self.embedding_ - np.mean(self.embedding_, axis=0)

        return self.embedding_

    def fit_transform(self, X, y=None, metric='euclidean'):
        return self._fit(X, y, metric)

    def fit(self, X, y=None, metric='euclidean'):
        return self._fit(X, y, metric)
