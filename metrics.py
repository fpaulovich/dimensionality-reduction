import math

import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import pdist

MACHINE_EPSILON = np.finfo(np.double).eps


def stress(X, y, metric='euclidean'):
    D_high = pdist(X, metric=metric)
    D_low = pdist(y, metric=metric)
    return math.sqrt(np.sum(((D_high - D_low) ** 2) / np.sum(D_high ** 2)))


def neighborhood_preservation(X, y, nr_neighbors=10, metric='euclidean'):
    dists_high, indexes_high = KDTree(X, leaf_size=2, metric=metric).query(X, k=nr_neighbors)
    dists_low, indexes_low = KDTree(y, leaf_size=2, metric=metric).query(y, k=nr_neighbors)

    neigh_pres = np.zeros(len(X))
    for i in range(len(X)):
        for p in range(nr_neighbors):
            for q in range(nr_neighbors):
                if indexes_high[i][p] == indexes_low[i][q]:
                    neigh_pres[i] = neigh_pres[i] + 1
        neigh_pres[i] = neigh_pres[i] / nr_neighbors

    return np.average(neigh_pres)


def neighborhood_hit(y, label, nr_neighbors=10, metric='euclidean'):
    dists_low, indexes_low = KDTree(y, leaf_size=2, metric=metric).query(y, k=nr_neighbors)

    neigh_hit = np.zeros(len(y))
    for i in range(len(y)):
        for j in range(nr_neighbors):
            if label[i] == label[indexes_low[i][j]]:
                neigh_hit[i] = neigh_hit[i] + 1
        neigh_hit[i] = neigh_hit[i] / nr_neighbors

    return np.average(neigh_hit)