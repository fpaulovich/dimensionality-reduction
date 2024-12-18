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
def move(ins1, distance_matrix, projection, learning_rate):
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
            delta = (d_original - dr2)  # * math.fabs(d_original - dr2)
            error += math.fabs(delta)

            # moving
            projection[ins2][0] += learning_rate * delta * (x1x2 / dr2)
            projection[ins2][1] += learning_rate * delta * (y1y2 / dr2)

    return error / size


@njit(parallel=False, fastmath=False)
def iteration(index, distance_matrix, projection, learning_rate):
    size = len(projection)
    error = 0

    for i in range(size):
        error += move(index[i], distance_matrix, projection, learning_rate)

    return error / size


def rotate(y, labels):
    size = len(y)

    # calculate clusters centroids
    labels_unique = list(set(labels))
    centroids = np.zeros((len(labels_unique), 2), dtype=np.float32)
    sizes = np.zeros(len(labels_unique), dtype=np.float32)

    labels_dict = {}
    for i in range(len(labels_unique)):
        labels_dict[labels_unique[i]] = i

    for i in range(size):
        centroids[labels_dict[labels[i]]] = centroids[labels_dict[labels[i]]] + y[i]
        sizes[labels_dict[labels[i]]] = sizes[labels_dict[labels[i]]] + 1

    for i in range(len(centroids)):
        centroids[i] = centroids[i] / sizes[i]

    # finding the farthest centroids
    max_dist = 0
    p0 = None  # centroid with lowest x value
    p1 = None  # centroid with largest x value
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            dist = np.linalg.norm(centroids[i]-centroids[j])

            if dist > max_dist:
                if centroids[i][0] < centroids[j][0]:
                    p0 = centroids[i]
                    p1 = centroids[j]
                else:
                    p0 = centroids[j]
                    p1 = centroids[i]

                max_dist = dist

    # move projection centroid to the origin
    y = y - p0

    # rotate
    v = (p1-p0) / np.linalg.norm(p1-p0)  # vector into the positive direction of x
    cos = v[0]
    sin = np.sign(v[1]) * math.sqrt(1 - cos*cos)

    for i in range(size):
        x_coord = cos * y[i][0] + sin * y[i][1]
        y_coord = -sin * y[i][0] + cos * y[i][1]
        y[i][0] = x_coord
        y[i][1] = y_coord

    # add projection centroid back
    # y = y + mean
    y = y - np.amin(y, axis=0)

    return y


# def compute_labels_order(y, labels):
#     labels_unique = list(set(labels))
#     centroids = np.zeros(len(labels_unique), dtype=np.float32)
#     instances_per_label = np.zeros(len(labels_unique), dtype=np.float32)
#
#     labels_dict = {}
#     for i in range(len(labels_unique)):
#         labels_dict[labels_unique[i]] = i
#
#     for i in range(len(labels)):
#         centroids[labels_unique[labels[i]]] = centroids[labels_unique[labels[i]]] + y[i][0]
#         instances_per_label[labels_unique[labels[i]]] = instances_per_label[labels_unique[labels[i]]] + 1
#
#     labels_centroids_dict = {}
#     for i in range(len(labels_unique)):
#         labels_centroids_dict[labels_unique[i]] = centroids[i] / instances_per_label[i]


def compute_positions(fixed_feature, feature_type):
    if feature_type == 'nominal':
        fixed_feature_unique = list(set(fixed_feature))
        positions = np.zeros(len(fixed_feature), dtype=np.float32)

        labels_dict = {}
        for i in range(len(fixed_feature_unique)):
            labels_dict[fixed_feature_unique[i]] = i + 1

        for i in range(len(fixed_feature)):
            positions[i] = (2 * labels_dict[fixed_feature[i]] - 1) / (2 * len(fixed_feature_unique))

        return positions
    elif feature_type == 'ordinal':
        min_val = min(fixed_feature)
        max_val = max(fixed_feature)

        # normalizing the fixed feature to [0,1]
        return (fixed_feature - min_val) / (max_val - min_val)
    else:
        return None


# calculate adaptively intervals according to the features values
def compute_intervals(positions, alpha):
    # remove the feature duplicate values and order
    fixed_feature_unique = list(set(positions))
    fixed_feature_unique = np.sort(fixed_feature_unique)

    # create index to contain intervals and add the first interval
    intervals_dict = {fixed_feature_unique[0]: [
        fixed_feature_unique[0] - alpha * (fixed_feature_unique[1] - fixed_feature_unique[0]) / 2,
        fixed_feature_unique[0] + alpha * (fixed_feature_unique[1] - fixed_feature_unique[0]) / 2
    ]}

    # create other intervals
    for i in range(1, len(fixed_feature_unique) - 1):
        intervals_dict[fixed_feature_unique[i]] = [
            fixed_feature_unique[i] - (alpha * (fixed_feature_unique[i] - fixed_feature_unique[i - 1]) / 2),
            fixed_feature_unique[i] + (alpha * (fixed_feature_unique[i + 1] - fixed_feature_unique[i]) / 2)
        ]

    # create last interval
    last = len(fixed_feature_unique) - 1
    intervals_dict[fixed_feature_unique[last]] = [
        fixed_feature_unique[last] - alpha * (fixed_feature_unique[last] - fixed_feature_unique[last - 1]) / 2,
        fixed_feature_unique[last] + alpha * (fixed_feature_unique[last] - fixed_feature_unique[last - 1]) / 2
    ]

    final_intervals = np.zeros((len(positions), 2), dtype=np.float32)

    for i in range(len(positions)):
        final_intervals[i][0] = intervals_dict[positions[i]][0]
        final_intervals[i][1] = intervals_dict[positions[i]][1]

    return final_intervals


def clipping(y, intervals):
    # pull
    for i in range(len(y)):
        min_int = intervals[i][0]
        max_int = intervals[i][1]
        y[i][0] = min_int if y[i][0] < min_int else (max_int if y[i][0] > max_int else y[i][0])

    return y


class DimenFixForceScheme:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95,
                 tolerance=0.00001,
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
        self.seed_ = seed
        self.n_components_ = n_components
        self.embedding_ = None
        self.fixed_feature_ = fixed_feature
        self.feature_type_ = feature_type
        self.alpha_ = alpha
        self.pulling_strategy_ = pulling_strategy
        self.iterations_to_pull_ = iterations_to_pull

    def _fit(self, X, y, distance_function):
        # create a distance matrix
        distance_matrix = create_distance_matrix(X, distance_function)
        distance_matrix = (distance_matrix - min(distance_matrix)) / (max(distance_matrix) - min(distance_matrix))
        size = len(X)

        # Processing the fixed feature
        if self.fixed_feature_ is None:
            raise ValueError('A feature to be fixed needs to be provided!')
        elif len(X) != len(self.fixed_feature_):
            raise ValueError('The dataset and the feature to be fixed must have the same sizes!')

        if self.n_components_ != 2:
            raise ValueError('Only 2 components supported for the reduction!')

        # compute positions
        positions = compute_positions(self.fixed_feature_, self.feature_type_)

        # compute intervals
        intervals = compute_intervals(positions, self.alpha_)

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

        # create random index
        index = np.random.permutation(size)

        # iterate until max_it or if the error does not change more than the tolerance
        error = math.inf
        for k in range(self.max_it_):
            learning_rate = self.learning_rate0_ * math.pow((1 - k / self.max_it_), self.decay_)
            new_error = iteration(index, distance_matrix, self.embedding_, learning_rate)

            # pull  (x-coordinate is fixed)
            if ((k+1) % self.iterations_to_pull_ == 0) or ((k-1) == self.max_it_):
                # rotate
                if self.feature_type_ == 'nominal':
                    self.embedding_ = rotate(self.embedding_, positions)

                # pull
                if self.pulling_strategy_ == 'clipping':
                    clipping(self.embedding_, intervals)



            # if math.fabs(new_error - error) < self.tolerance_:
            #     break

            error = new_error

        # setting the min to (0,0)
        # self.embedding_ = self.embedding_ - np.amin(self.embedding_, axis=0)

        return self.embedding_

    def fit_transform(self, X, y=None, distance_function=distance.euclidean):
        return self._fit(X, y, distance_function)

    def fit(self, X, y=None, distance_function=distance.euclidean):
        return self._fit(X, y, distance_function)
