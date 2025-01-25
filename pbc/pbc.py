# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

# If you use this implementation, please cite
# F. V. Paulovich and R. Minghim, "Text Map Explorer: a Tool to Create and Explore Document Maps," Tenth International
# Conference on Information Visualisation (IV'06), London, UK, 2006, pp. 245-251, doi: 10.1109/IV.2006.104.

import numpy as np
import math

from sklearn.cluster import KMeans
from force.force_scheme import ForceScheme


class PBC:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95,
                 tolerance=0.00001,
                 seed=7,
                 n_components=2,
                 cluster_factor=1):

        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.tolerance_ = tolerance
        self.seed_ = seed
        self.n_components_ = n_components
        self.cluster_factor_ = cluster_factor
        self.embedding_ = None

    def _fit(self, X, metric):
        # create clusters
        n_clusters = int(math.sqrt(len(X)))
        kmeans = KMeans(n_clusters=n_clusters,
                        init='k-means++',
                        algorithm='lloyd',
                        random_state=self.seed_,
                        n_init="auto").fit(X)

        # create indexes for the clusters
        cluster_indexes = {}
        for i in range(n_clusters):
            cluster_indexes[i] = []
        for i in range(len(kmeans.labels_)):
            cluster_indexes[kmeans.labels_[i]].append(i)

        # calculate cluster centers
        cluster_centers = np.empty((n_clusters, len(X[0])))
        for i in range(n_clusters):
            cluster_centers[i] = np.mean(X[cluster_indexes[i], :], axis=0)

        # project cluster centers
        cluster_centers_projection = ForceScheme(max_it=self.max_it_,
                                                 learning_rate0=self.learning_rate0_,
                                                 decay=self.decay_,
                                                 tolerance=self.tolerance_,
                                                 seed=self.seed_,
                                                 n_components=self.n_components_
                                                 ).fit_transform(cluster_centers, metric=metric)

        # create final embedding
        self.embedding_ = np.zeros((len(X), self.n_components_))
        for i in range(n_clusters):
            # project each cluster individually
            cluster_data = X[cluster_indexes[i], :]
            cluster_projection = ForceScheme(max_it=self.max_it_,
                                             learning_rate0=self.learning_rate0_,
                                             decay=self.decay_,
                                             tolerance=self.tolerance_,
                                             seed=self.seed_,
                                             n_components=self.n_components_
                                             ).fit_transform(cluster_data, metric=metric)

            # center the cluster projection
            cluster_projection = np.subtract(cluster_projection, np.mean(cluster_projection, axis=0))

            # scale the cluster projection by the cluster factor
            cluster_projection = cluster_projection * self.cluster_factor_

            # move the center of the cluster projection to the cluster center position
            cluster_projection = np.add(cluster_projection, cluster_centers_projection[i])

            for j in range(len(cluster_projection)):
                self.embedding_[cluster_indexes[i][j]] = cluster_projection[j]

        return self.embedding_

    def fit_transform(self, X, y=None, metric='euclidean'):
        return self._fit(X, metric)

    def fit(self, X, y=None, metric='euclidean'):
        return self._fit(X, metric)
