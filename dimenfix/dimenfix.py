import numpy as np
import math

from log import print_layout

from numpy import random
from numba import njit, prange
from scipy.spatial import distance

PULLING_TYPES = {'clipping', 'gaussian', 'rescale'}
DATA_TYPES = {'nominal', 'numeric_ordinal'}


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
            dist = np.linalg.norm(centroids[i] - centroids[j])

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
    v = (p1 - p0) / np.linalg.norm(p1 - p0)  # vector into the positive direction of x
    cos = v[0]
    sin = np.sign(v[1]) * math.sqrt(1 - cos * cos)

    for i in range(size):
        x_coord = cos * y[i][0] + sin * y[i][1]
        y_coord = -sin * y[i][0] + cos * y[i][1]
        y[i][0] = x_coord
        y[i][1] = y_coord

    # add projection centroid back
    # y = y + mean
    y = y - np.amin(y, axis=0)

    return y


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
    elif feature_type == 'numeric_ordinal':
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


def clipping_pull(y, intervals):
    for i in range(len(y)):
        min_int = intervals[i][0]
        max_int = intervals[i][1]
        y[i][0] = min_int if y[i][0] < min_int else (max_int if y[i][0] > max_int else y[i][0])
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


class DimenFix:

    def __init__(self,
                 feature_type,
                 pulling_strategy,
                 alpha=1.0):

        self.feature_type_ = feature_type
        self.pulling_strategy_ = pulling_strategy
        self.alpha_ = alpha
        self.positions_ = None
        self.intervals_ = None

    def fit(self, fixed_feature):
        if self.feature_type_ not in DATA_TYPES:
            raise ValueError("results: feature_type must be one of %r." % DATA_TYPES)

        if self.pulling_strategy_ not in PULLING_TYPES:
            raise ValueError("results: pulling_strategy must be one of %r." % PULLING_TYPES)

        # compute positions
        self.positions_ = compute_positions(fixed_feature, self.feature_type_)

        # compute intervals
        self.intervals_ = compute_intervals(self.positions_, self.alpha_)

        return self

    def transform(self, embedding):
        # rotate
        if self.feature_type_ == 'nominal':
            embedding = rotate(embedding, self.positions_)

            print_layout(embedding, self.positions_, title="after dimenfix (rotate)")

        # pull
        if self.pulling_strategy_ == 'clipping':
            embedding = clipping_pull(embedding, self.intervals_)
        elif self.pulling_strategy_ == 'gaussian':
            print('gaussian')
        elif self.pulling_strategy_ == 'rescale':
            print('rescale')

        print_layout(embedding, self.positions_, title="after dimenfix (pull)")

        return embedding

