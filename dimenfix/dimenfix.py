# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import numpy as np
import math
import scipy.stats as st

from log import print_layout, clean

from sklearn.preprocessing import LabelEncoder

active_log = False

PULLING_TYPES = {'clipping', 'gaussian', 'rescale'}
DATA_TYPES = {'nominal', 'ordinal'}


def split_groups(label):
    groups = []

    # find unique labels
    labels_unique = list(set(label))

    labels_dict = {}
    for i in range(len(labels_unique)):
        labels_dict[labels_unique[i]] = i
        groups.append([])

    for i in range(len(label)):
        groups[labels_dict[label[i]]].append(i)

    return groups


def calculate_centroids(embedding, groups):
    nr_centroids = len(groups)
    centroids = np.zeros((nr_centroids, embedding.shape[1]), dtype=np.float32)

    for i in range(nr_centroids):
        for j in range(len(groups[i])):
            centroids[i] = centroids[i] + embedding[groups[i][j]]
        centroids[i] = centroids[i] / len(groups[i])

    return centroids


def compute_positions_nominal(embedding, groups):
    centroids = calculate_centroids(embedding, groups)

    groups_centroids = []
    for i in range(len(centroids)):
        groups_centroids.append([i, centroids[i][0], centroids[i][1]])

    # order centroids considering their x-coordinates
    groups_centroids.sort(key=lambda x: x[1])
    centroids_order = []
    for i in range(len(centroids)):
        centroids_order.append(groups_centroids[i][0])

    positions = np.zeros(len(embedding), dtype=np.float32)

    for i in range(len(centroids)):
        for j in range(len(groups[i])):
            positions[groups[i][j]] = (2 * centroids_order.index(i) + 1) / (2 * len(centroids))

    return positions


def rotate(embedding, groups):
    size = len(embedding)

    # calculate centroids
    centroids = calculate_centroids(embedding, groups)

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
    embedding = embedding - p0

    # rotate
    v = (p1 - p0) / np.linalg.norm(p1 - p0)  # vector into the positive direction of x
    cos = v[0]
    sin = np.sign(v[1]) * math.sqrt(1 - cos * cos)

    for i in range(size):
        x_coord = cos * embedding[i][0] + sin * embedding[i][1]
        y_coord = -sin * embedding[i][0] + cos * embedding[i][1]
        embedding[i][0] = x_coord
        embedding[i][1] = y_coord

    # add projection centroid back
    # y = y + mean
    embedding = embedding - np.amin(embedding, axis=0)

    return embedding


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


def clipping_pull(embedding, intervals):
    for i in range(len(embedding)):
        min_int = intervals[i][0]
        max_int = intervals[i][1]
        embedding[i][0] = min_int if embedding[i][0] < min_int else \
            (max_int if embedding[i][0] > max_int else embedding[i][0])
    return embedding


def rescale_pull(embedding, groups, intervals):
    for group in groups:
        # getting the max and min x-coordinate in the projection
        min_x = embedding[group[0]][0]
        max_x = embedding[group[0]][0]

        for index in group:
            if embedding[index][0] > max_x:
                max_x = embedding[index][0]
            elif embedding[index][0] < min_x:
                min_x = embedding[index][0]

        # getting the min and max range
        min_range = intervals[group[0]][0]
        max_range = intervals[group[0]][1]

        for index in group:
            embedding[index][0] = (((embedding[index][0] - min_x) / (max_x - min_x)) *
                                   (max_range - min_range)) + min_range

    return embedding


def gaussian_pull(embedding, positions, intervals):
    confidence_interval = 0.45
    z_score = st.norm.ppf((1 - confidence_interval) / 2)

    for i in range(len(embedding)):
        min_int = intervals[i][0]
        max_int = intervals[i][1]

        gaussian_range = (math.fabs(positions[i] - min_int) if embedding[i][0] < positions[i]
                          else math.fabs(max_int - positions[i]))
        sigma = gaussian_range / z_score

        axis_difference = embedding[i][0] - positions[i]
        moving_to_ratio = math.exp(-(axis_difference ** 2 / (2 * (sigma ** 2))))
        embedding[i][0] = positions[i] + (moving_to_ratio * axis_difference)
    return embedding


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
        self.groups_ = None

    def fit(self, embedding, fixed_feature):
        if self.feature_type_ not in DATA_TYPES:
            raise ValueError("results: feature_type must be one of %r." % DATA_TYPES)

        if self.pulling_strategy_ not in PULLING_TYPES:
            raise ValueError("results: pulling_strategy must be one of %r." % PULLING_TYPES)

        if self.pulling_strategy_ == 'rescale' and self.feature_type_ == 'ordinal':
            raise ValueError("results: pulling_strategy rescale cannot be used with feature_type ordinal.")

        # if feature is nominal, create groups of instances based on the fixed_feature
        if self.feature_type_ == 'nominal':
            fixed_feature = LabelEncoder().fit_transform(fixed_feature)
            self.groups_ = split_groups(fixed_feature)

        # compute positions
        if self.feature_type_ == 'nominal':
            self.positions_ = compute_positions_nominal(embedding, self.groups_)
        elif self.feature_type_ == 'ordinal':
            self.positions_ = fixed_feature

        # compute intervals
        self.intervals_ = compute_intervals(self.positions_, self.alpha_)

        clean(active=active_log)

        return self

    def transform(self, embedding):
        print_layout(embedding, self.positions_, title="before dimenfix", active=active_log)

        # rotate and calculate labels position
        if self.feature_type_ == 'nominal':
            embedding = rotate(embedding, self.groups_)

            print_layout(embedding, self.positions_, title="after dimenfix (rotate)", active=active_log)

            self.positions_ = compute_positions_nominal(embedding, self.groups_)
            self.intervals_ = compute_intervals(self.positions_, self.alpha_)

        # center embedding
        embedding_mean = np.mean(embedding, axis=0)
        embedding = embedding - embedding_mean

        # scale embedding to match the intervals
        min_interval = np.min(self.intervals_)
        max_interval = np.max(self.intervals_)
        min_fix_coord = np.min(embedding, axis=0)[0]
        max_fix_coord = np.max(embedding, axis=0)[0]
        scale = (max_interval - min_interval) / (max_fix_coord - min_fix_coord)
        embedding = (embedding * scale) + ((max_interval + min_interval) / 2)

        # pull
        if self.pulling_strategy_ == 'clipping':
            embedding = clipping_pull(embedding, self.intervals_)
        elif self.pulling_strategy_ == 'gaussian':
            embedding = gaussian_pull(embedding, self.positions_, self.intervals_)
        elif self.pulling_strategy_ == 'rescale':
            embedding = rescale_pull(embedding, self.groups_, self.intervals_)

        # restore to original scale if data is nominal
        if self.feature_type_ == 'nominal':
            embedding = embedding - np.mean(embedding, axis=0)
            embedding = (embedding / scale) + embedding_mean

        print_layout(embedding, self.positions_, title="after dimenfix (pull)", active=active_log)

        return embedding
