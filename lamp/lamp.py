# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT
# code inspired by https://github.com/lgnonato/LAMP/blob/master/lamp.py

# If you use this implementation, please cite
# P. Joia, D. Coimbra, J. A. Cuminato, F. V. Paulovich and L. G. Nonato, "Local Affine Multidimensional Projection,"
# in IEEE Transactions on Visualization and Computer Graphics, vol. 17, no. 12, pp. 2563-2571, Dec. 2011,
# doi: 10.1109/TVCG.2011.220.

import numpy as np
from force.force_scheme import ForceScheme
from sklearn.neighbors import KDTree

epsilon = 1e-7


def orthogonal_mapping(X, X_sample_, y_sample_, embedding, n_components, nr_neighbors, metric):
    tree = KDTree(X_sample_, leaf_size=2, metric=metric)
    dists, indexes = tree.query(X, k=nr_neighbors)
    weights = 1.0 / (dists + epsilon)

    for i in range(len(X)):
        if dists[i][0] < epsilon:
            embedding[i] = y_sample_[indexes[i][0]]
        else:
            # copy only data of the nearest neighbors
            sample_data = X_sample_[indexes[i], :]
            sample_embedding = y_sample_[indexes[i], :]

            alpha = np.sum(weights[i])
            x_tilde = np.dot(sample_data.T, weights[i].T) / alpha
            y_tilde = np.dot(sample_embedding.T, weights[i].T) / alpha
            x_hat = sample_data - x_tilde
            y_hat = sample_embedding - y_tilde
            D = np.diag(np.sqrt(weights[i]))
            A = np.dot(D, x_hat)
            B = np.dot(D, y_hat)
            U, s, V = np.linalg.svd(np.dot(A.T, B), full_matrices=True)
            M = np.dot(U[:, :n_components], V)
            embedding[i] = np.dot((X[i] - x_tilde), M) + y_tilde


class Lamp:

    def __init__(self,
                 nr_neighbors=0,
                 n_components=2):
        self.nr_neighbors_ = nr_neighbors
        self.n_components_ = n_components
        self.sample_size_ = 0
        self.X_sample_ = None
        self.y_sample_ = None
        self.embedding_ = None

    def _fit(self, X_sample, y_sample, metric):
        # use all samples to project the data if nr_neighbors is not provided
        if self.nr_neighbors_ == 0:
            self.nr_neighbors_ = len(X_sample)

        if len(X_sample) < self.nr_neighbors_:
            raise ValueError('The X_sample needs to be larger than the number of neighbors.')

        self.X_sample_ = X_sample
        self.sample_size_ = len(self.X_sample_)

        # if sample projection is not provided, project using ForceScheme
        if y_sample is None:
            self.y_sample_ = ForceScheme(max_it=100,
                                         n_components=self.n_components_
                                         ).fit_transform(self.X_sample_, metric=metric)
        else:
            if len(X_sample) != len(y_sample):
                raise ValueError('The X_sample and y_sample sizes needs to be the same.')
            self.y_sample_ = y_sample
            self.n_components_ = len(self.y_sample_[0])

        # centering sample data and projection
        self.y_sample_center_ = np.average(self.y_sample_, axis=0)
        self.y_sample_ = np.subtract(self.y_sample_, self.y_sample_center_)

        self.X_sample_center_ = np.average(self.X_sample_, axis=0)
        self.X_sample_ = np.subtract(self.X_sample_, self.X_sample_center_)

        return self

    def transform(self, X, metric='euclidean'):
        self.embedding_ = np.zeros((len(X), self.n_components_))

        # centering the data using the sample data center
        X = np.subtract(X, self.X_sample_center_)

        # orthogonally map the new data X
        orthogonal_mapping(X, self.X_sample_, self.y_sample_, self.embedding_,
                           self.n_components_, self.nr_neighbors_, metric)

        # adding the center of the sample projection back
        return np.add(self.embedding_, self.y_sample_center_)

    def fit_transform(self, X, X_sample, y_sample=None, metric='euclidean'):
        self._fit(X_sample, y_sample, metric)
        return self.transform(X, metric)

    def fit(self, X_sample, y_sample=None, metric='euclidean'):
        return self._fit(X_sample, y_sample, metric)
