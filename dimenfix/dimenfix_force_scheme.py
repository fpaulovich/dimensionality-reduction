# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import numpy as np
import math

from numpy import random
from numba import njit, prange
from scipy.spatial import distance

from dimenfix import DimenFix


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

    for i in prange(size):
        for j in range(i, size):
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


@njit(parallel=False, fastmath=False)
def iteration(index, distance_matrix, projection, learning_rate, n_components):
    size = len(projection)
    error = 0

    for i in range(size):
        if n_components == 2:
            error += move2D(index[i], distance_matrix, projection, learning_rate)
        else:
            error += move(index[i], distance_matrix, projection, learning_rate)

    return error / size


class DimenFixForceScheme:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95,
                 tolerance=0.00001,
                 n_iter_without_progress=10,
                 seed=7,
                 n_components=2,
                 fixed_feature=None,
                 feature_type='ordinal',
                 alpha=1.0,
                 pulling_strategy='clipping',
                 iterations_to_pull=10):

        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.tolerance_ = tolerance
        self.n_iter_without_progress_ = n_iter_without_progress
        self.seed_ = seed
        self.n_components_ = n_components
        self.embedding_ = None
        self.fixed_feature_ = fixed_feature
        self.feature_type_ = feature_type
        self.alpha_ = alpha
        self.pulling_strategy_ = pulling_strategy
        self.iterations_to_pull_ = iterations_to_pull

    def _fit(self, X, y, metric):
        # create a distance matrix
        distance_matrix = create_distance_matrix(X, metric)
        # distance_matrix = (distance_matrix - min(distance_matrix)) / (max(distance_matrix) - min(distance_matrix))
        size = len(X)

        # Processing the fixed feature
        if self.fixed_feature_ is None:
            raise ValueError('A feature to be fixed needs to be provided!')
        elif len(X) != len(self.fixed_feature_):
            raise ValueError('The dataset and the feature to be fixed must have the same sizes!')

        # set the random seed
        np.random.seed(self.seed_)

        # randomly initialize the projection
        if y is None:
            self.embedding_ = np.random.random((size, self.n_components_))
        else:
            if np.shape(y)[1] == self.n_components_:
                self.embedding_ = y
            else:
                raise ValueError('The n_components should be equal to the number of dimensions of the input embedding!')

        # init DimenFix
        dfix = DimenFix(feature_type=self.feature_type_,
                        pulling_strategy=self.pulling_strategy_,
                        alpha=self.alpha_).fit(self.embedding_, self.fixed_feature_)

        # create random index
        index = np.random.permutation(size)

        # iterate until max_it or if the error does not change more than the tolerance
        iter_without_progress = 0
        error = math.inf
        for k in range(self.max_it_):
            learning_rate = self.learning_rate0_ * math.pow((1 - k / self.max_it_), self.decay_)
            new_error = iteration(index, distance_matrix, self.embedding_, learning_rate, self.n_components_)

            # pull  (x-coordinate is fixed)
            if (k + 1) % self.iterations_to_pull_ == 0:
                self.embedding_ = dfix.transform(self.embedding_)

            if math.fabs(new_error - error) < self.tolerance_:
                iter_without_progress = iter_without_progress + 1
            else:
                iter_without_progress = 0

            if iter_without_progress >= self.n_iter_without_progress_:
                break

            error = new_error

        self.embedding_ = dfix.transform(self.embedding_)

        return self.embedding_

    def fit_transform(self, X, y=None, metric='euclidean'):
        return self._fit(X, y, metric)

    def fit(self, X, y=None, metric='euclidean'):
        return self._fit(X, y, metric)
