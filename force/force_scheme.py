import numpy as np
import math

from numpy import random
from numba import njit, prange
from scipy.spatial import distance


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

            # getting te index in the distance matrix and getting the value
            r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(i,j)
            s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(i,j)
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

            # getting te index in the distance matrix and getting the value
            r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(i,j)
            s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(i,j)
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
        ins1 = index[i]

        if n_components == 2:
            error += move2D(ins1, distance_matrix, projection, learning_rate)
        else:
            error += move(ins1, distance_matrix, projection, learning_rate)

    return error / size


class ForceScheme:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95,
                 tolerance=0.00001,
                 seed=7,
                 n_components=2):

        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.tolerance_ = tolerance
        self.seed_ = seed
        self.n_components_ = n_components
        self.embedding_ = None

    def _fit(self, X, y, distance_function):
        # create a distance matrix
        distance_matrix = create_distance_matrix(X, distance_function)
        size = len(X)

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

        # create random index
        index = np.random.permutation(size)

        # iterate until max_it or if the error does not change more than the tolerance
        error = math.inf
        for k in range(self.max_it_):
            learning_rate = self.learning_rate0_ * math.pow((1 - k / self.max_it_), self.decay_)
            new_error = iteration(index, distance_matrix, self.embedding_, learning_rate, self.n_components_)

            if math.fabs(new_error - error) < self.tolerance_:
                break

            error = new_error

        # setting the min to (0,0)
        self.embedding_ = self.embedding_ - np.amin(self.embedding_, axis=0)

        return self.embedding_

    def fit_transform(self, X, y=None, distance_function=distance.euclidean):
        return self._fit(X, y, distance_function)

    def fit(self, X, y=None, distance_function=distance.euclidean):
        return self._fit(X, y, distance_function)
