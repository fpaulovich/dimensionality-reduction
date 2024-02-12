import numpy as np
import math

from numpy import random
from numba import njit, prange
from scipy.spatial import distance

from sklearn.cluster import KMeans

import heapq


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


def calculate_k_distance(distance_matrix, nr_neighbors):
    total = len(distance_matrix)
    size = int((math.sqrt(1 + 8 * total) - 1) / 2)

    # adjusting the number of neighbors in case it is larger than the dataset
    nr_neighbors = min(nr_neighbors, size - 1)

    k_distance = np.zeros(size, dtype=np.float32)

    for i in range(size):
        heap = []

        for j in range(size):
            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            if i != j:
                heapq.heappush(heap, drn)

        for k in range(nr_neighbors - 1):
            heapq.heappop(heap)

        k_distance[i] = heapq.heappop(heap)

    return k_distance


def calculate_k_nn_graph(distance_matrix, nr_neighbors):
    total = len(distance_matrix)
    size = int((math.sqrt(1 + 8 * total) - 1) / 2)

    # adjusting the number of neighbors in case it is larger than the dataset
    nr_neighbors = min(nr_neighbors, size - 1)

    # k_nn_graph =  np.zeros((size, nr_neighbors), dtype=np.float32)
    k_nn_graph = [[] for i in range(size)]

    for i in range(size):
        heap = []

        for j in range(size):
            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            if i != j:
                heapq.heappush(heap, (drn, j))

        for k in range(nr_neighbors):
            item = heapq.heappop(heap)
            k_nn_graph[i].append([item[1], item[0]])
            k_nn_graph[item[1]].append([i, item[0]])

        print(len(k_nn_graph[i]))

    return k_nn_graph


def sanity_check(distance_matrix, avg_k_distance):
    total = len(distance_matrix)
    size = int((math.sqrt(1 + 8 * total) - 1) / 2)

    for i in range(size):
        count = 0

        for j in range(size):
            r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
            s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
            drn = distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]

            if drn <= avg_k_distance:
                count = count + 1

        print(count)


@njit(parallel=True, fastmath=False)
def move_local(ins1, k_distances, distance_matrix, projection, learning_rate, prob_threshold, max_dist):
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

            prob = np.random.random()

            if drn <= k_distances[ins2]:
                # calculate the movement
                delta = (drn - dr2)
                error += math.fabs(delta)

                # moving
                projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
                projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)
            elif prob < prob_threshold:
                # calculate the movement
                delta = (drn + (5*np.random.random()) * max_dist - dr2)
                error += math.fabs(delta)

                # moving
                projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
                projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)

    return error / size


@njit(parallel=True, fastmath=False)
def move_original(ins1, distance_matrix, projection, learning_rate):
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

            # moving
            projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
            projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)

    return error / size


@njit(parallel=False, fastmath=False)
def iteration_local(index, k_distances, distance_matrix, projection, learning_rate, prob_threshold, max_dist):
    size = len(projection)
    error = 0

    for i in range(size):
        ins1 = index[i]
        error += move_local(ins1, k_distances, distance_matrix, projection, learning_rate, prob_threshold, max_dist)

    return error / size


@njit(parallel=False, fastmath=False)
def iteration_original(index, distance_matrix, projection, learning_rate):
    size = len(projection)
    error = 0

    for i in range(size):
        ins1 = index[i]
        error += move_original(ins1, distance_matrix, projection, learning_rate)

    return error / size


class LocalForceScheme:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95,
                 tolerance=0.00001,
                 seed=7,
                 prob_threshold=0.1,
                 nr_neighbors=10):

        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.tolerance_ = tolerance
        self.seed_ = seed
        self.prob_threshold_ = prob_threshold
        self.nr_neighbors_ = nr_neighbors
        self.embedding_ = None

    def _fit(self, X, y, distance_function):
        # create a distance matrix
        distance_matrix = create_distance_matrix(X, distance_function)
        size = len(X)

        k_distances = calculate_k_distance(distance_matrix, self.nr_neighbors_)
        # k_distances.fill(np.median(k_distances))
        # sanity_check(distance_matrix, avg_k_distance)

        # k_nn_graph = calculate_k_nn_graph(distance_matrix, self.nr_neighbors_)

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
            learning_rate = self.learning_rate0_ * math.pow((1 - k / global_it), self.decay_)
            new_error = iteration_original(index, distance_matrix, self.embedding_, learning_rate)

            if math.fabs(new_error - error) < self.tolerance_:
                print('nr iterations global: ', k)
                break

            error = new_error

        # iterate until max_it or if the error does not change more than the tolerance
        error = math.inf
        max_dist = max(distance_matrix)
        for k in range(self.max_it_):
            learning_rate = self.learning_rate0_ * math.pow((1 - k / self.max_it_), self.decay_)
            new_error = iteration_local(index, k_distances, distance_matrix, self.embedding_, learning_rate,
                                        self.prob_threshold_, max_dist)

            if math.fabs(new_error - error) < self.tolerance_:
                print('nr iterations local: ', k)
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
