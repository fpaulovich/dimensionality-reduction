import numpy as np
import math

from numpy import random
from numba import njit, prange
from scipy.spatial import distance

from sklearn.cluster import KMeans


# @njit(parallel=False, fastmath=False)
def create_distance_matrix(X, distance_function):
    size = len(X)
    distance_matrix = np.zeros(int(size * (size + 1) / 2), dtype=np.float32)

    k = 0
    for i in range(size):
        for j in range(i, size):
            distance_matrix[k] = distance_function(X[i], X[j])
            k = k + 1

    return distance_matrix


def calculate_weight_forces(X, distance_function, random_state):
    size = len(X)
    weights = np.zeros(size * size, dtype=np.float32)

    k_means = KMeans(init='k-means++', random_state=random_state, algorithm='lloyd',
                     n_clusters=int(math.sqrt(size)))
    k_means.fit(X)

    for i in range(size):
        for j in range(size):
            if k_means.labels_[i] == k_means.labels_[j]:
                weights[int(i * size + j)] = -1
            else:
                weights[int(i * size + j)] = distance_function(k_means.cluster_centers_[k_means.labels_[i]],
                                                               k_means.cluster_centers_[k_means.labels_[j]])
                # weights[int(i * size + j)] = distance_function(X[i], X[j])

    return weights


@njit(parallel=True, fastmath=False)
def move(ins1, weights, distance_matrix, projection, learning_rate, cluster_factor):
    size = len(projection)
    total = len(distance_matrix)
    error = 0

    for ins2 in prange(size):
        if ins1 != ins2:
            x1x2 = projection[ins2][0] - projection[ins1][0]
            y1y2 = projection[ins2][1] - projection[ins1][1]
            dr2 = max(math.sqrt(x1x2 * x1x2 + y1y2 * y1y2), 0.0001)

            # getting te index in the distance matrix and getting the value
            r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(i,j)
            s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            # calculate the movement
            delta = (drn - dr2)
            error += math.fabs(delta)

            # weighting
            weight = weights[int(ins1 * size + ins2)]
            delta = delta if weight == -1 else delta + (weight * cluster_factor)

            # moving
            projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
            projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)

    return error / size


@njit(parallel=False, fastmath=False)
def iteration(index, weights, distance_matrix, projection, learning_rate, cluster_factor):
    size = len(projection)
    error = 0

    for i in range(size):
        ins1 = index[i]
        error += move(ins1, weights, distance_matrix, projection, learning_rate, cluster_factor)

    return error / size


class LocalForceScheme:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95,
                 tolerance=0.00001,
                 seed=7,
                 cluster_factor=2):

        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.tolerance_ = tolerance
        self.seed_ = seed
        self.cluster_factor_ = cluster_factor
        self.embedding_ = None

    def _fit(self, X, y, distance_function):
        # create a distance matrix
        distance_matrix = create_distance_matrix(X, distance_function)
        size = len(X)

        # calculate the weights for the forces
        weights = calculate_weight_forces(X, distance_function, self.seed_)

        # set the random seed
        np.random.seed(self.seed_)

        # randomly initialize the projection
        if y is None:
            self.embedding_ = np.random.random((size, 2))
        else:
            self.embedding_ = y

        # create random index
        index = np.random.permutation(size)

        # iterate until max_it or if the error does not change more than the tolerance
        error = math.inf
        for k in range(self.max_it_):
            learning_rate = self.learning_rate0_ * math.pow((1 - k / self.max_it_), self.decay_)
            new_error = iteration(index, weights, distance_matrix, self.embedding_, learning_rate, self.cluster_factor_)

            if math.fabs(new_error - error) < self.tolerance_:
                break

            error = new_error

        # setting the min to (0,0)
        min_x = min(self.embedding_[:, 0])
        min_y = min(self.embedding_[:, 1])
        for i in range(size):
            self.embedding_[i][0] -= min_x
            self.embedding_[i][1] -= min_y

        return self.embedding_

    def fit_transform(self, X, y=None, distance_function=distance.euclidean):
        return self._fit(X, y, distance_function)

    def fit(self, X, y=None, distance_function=distance.euclidean):
        return self._fit(X, y, distance_function)
