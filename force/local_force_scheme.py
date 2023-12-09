import numpy as np
import math

from numpy import random
from numba import njit, prange, jit
from scipy.spatial import distance
from utils.min_k_inverted import calculate_k_distance
from utils.distance import euclidean_distance_numba, get_distance
from utils.sort import min_k_numba
from force_scheme import move as move_original, iteration as iteration_original
from sklearn.cluster import KMeans


@njit(fastmath=False)
def ij_to_matrix(i, j, size):
    return (i * size) + (j - i)


@njit(fastmath=False)
def size_to_matrix_size(size):
    return int(size * (size + 1) / 2)


# @njit(parallel=True, fastmath=False)
def create_distance_matrix(X):
    size = len(X)
    distance_matrix = np.zeros(int(size * (size + 1) / 2), dtype=np.float32)

    k = 0
    for i in range(size):
        for j in range(i, size):
            distance_matrix[k] = euclidean_distance_numba(X[i], X[j])
            k = k + 1

    return distance_matrix


def sanity_check(distance_matrix, avg_k_distance):
    total = len(distance_matrix)
    size = int((math.sqrt(1 + 8 * total) - 1) / 2)

    for i in range(size):
        count = 0

        for j in range(size):
            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[
                int(total - ((size - r) * (size - r + 1) / 2) + (s - r))
            ]

            if drn <= avg_k_distance:
                count = count + 1

        print(count)


@njit(parallel=False, fastmath=False)
def move_local(
    ins1,
    k_distances_idx,
    k_distance,
    distance_matrix,
    projection,
    learning_rate,
    prob_threshold,
    max_dist,
    verbose,
):
    size = len(projection)
    total = len(distance_matrix)
    error = 0

    # for ins2 in prange(size):
    for i in range(k_distances_idx.shape[1]):
        ins2 = k_distances_idx[ins1, i]
        if ins2 == -1:
            break
        # if ins1 == ins2:
        # continue

        x1x2 = projection[ins2][0] - projection[ins1][0]
        y1y2 = projection[ins2][1] - projection[ins1][1]
        dr2 = max(math.sqrt(x1x2 * x1x2 + y1y2 * y1y2), 0.0001)

        # getting the index in the distance matrix and getting the value
        drn = get_distance(distance_matrix, ins1, ins2, total, size)

        # calculate the movement
        delta = drn - dr2
        error += math.fabs(delta)

        # moving
        projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
        projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)

    rand_idx = np.random.choice(int(size), int(size * prob_threshold))
    for i in range(rand_idx.shape[0]):
        ins2 = rand_idx[i]
        if ins1 == ins2 or ins2 in k_distances_idx[ins1]:
            continue

        x1x2 = projection[ins2][0] - projection[ins1][0]
        y1y2 = projection[ins2][1] - projection[ins1][1]
        dr2 = max(math.sqrt(x1x2 * x1x2 + y1y2 * y1y2), 0.0001)

        # getting te index in the distance matrix and getting the value
        drn = get_distance(distance_matrix, ins1, ins2, total, size)

        # calculate the movement
        delta = drn + (5 * np.random.random()) * max_dist - dr2
        error += math.fabs(delta)

        # moving
        projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
        projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)

    return error / size


@njit(parallel=False, fastmath=False)
def iteration_local(
    index,
    k_distances_idx,
    k_distance,
    distance_matrix,
    projection,
    learning_rate,
    prob_threshold,
    max_dist,
    verbose,
):
    size = len(projection)
    error = 0

    for i in range(size):
        ins1 = index[i]
        error += move_local(
            ins1,
            k_distances_idx,
            k_distance,
            distance_matrix,
            projection,
            learning_rate,
            prob_threshold,
            max_dist,
            verbose,
        )

    return error / size


class LocalForceScheme:
    def __init__(
        self,
        max_it=100,
        learning_rate0=0.5,
        decay=0.95,
        tolerance=0.00001,
        seed=7,
        prob_threshold=0.1,
        nr_neighbors=10,
        verbose=False,
    ):
        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.tolerance_ = tolerance
        self.seed_ = seed
        self.prob_threshold_ = prob_threshold
        self.nr_neighbors_ = nr_neighbors
        self.embedding_ = None
        self.verbose = verbose

    def _fit(self, X, y, distance_function):
        # create a distance matrix
        distance_matrix = create_distance_matrix(X)
        size = len(X)

        if self.verbose: print("distance matrix created")
        k_distances_idx, k_distance = calculate_k_distance(
            distance_matrix, self.nr_neighbors_
        )
        if self.verbose: print("k distances calculated")
        # k_distances.fill(np.median(k_distances))
        # sanity_check(distance_matrix, avg_k_distance)

        # set the random seed
        np.random.seed(self.seed_)

        # randomly initialize the projection
        if y is None:
            self.embedding_ = np.random.random((size, 2))
        else:
            self.embedding_ = y

        # create random index
        index = np.random.permutation(size)

        # iterate some original moves
        error = math.inf
        global_it = 50
        for k in range(global_it):
            learning_rate = self.learning_rate0_ * math.pow(
                (1 - k / global_it), self.decay_
            )
            new_error = iteration_original(
                index, distance_matrix, self.embedding_, learning_rate
            )

            if math.fabs(new_error - error) < self.tolerance_:
                print("nr iterations global: ", k)
                break

            error = new_error

        # iterate until max_it or if the error does not change more than the tolerance
        error = math.inf
        max_dist = max(distance_matrix)
        for k in range(self.max_it_):
            if self.verbose:
                print("iteration: ", k + 1, error)
            learning_rate = self.learning_rate0_ * math.pow(
                (1 - k / self.max_it_), self.decay_
            )
            new_error = iteration_local(
                index,
                k_distances_idx,
                k_distance,
                distance_matrix,
                self.embedding_,
                learning_rate,
                self.prob_threshold_,
                max_dist,
                self.verbose,
            )

            if math.fabs(new_error - error) < self.tolerance_:
                if self.verbose:
                    print("nr iterations local: ", k)
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
