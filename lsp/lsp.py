# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright 2024 Fernando V. Paulovich
# License: BSD-3-Clause
# code inspired by https://github.com/lgnonato/LAMP/blob/master/lamp.py

# If you use this implementation, please cite
# Paulovich FV, Nonato LG, Minghim R, Levkowitz H. Least square projection: a fast high-precision multidimensional
# projection technique and its application to document mapping. IEEE Trans Vis Comput Graph. 2008 May-Jun;14(3):564-75.
# doi: 10.1109/TVCG.2007.70443.

import numpy as np
from force.force_scheme import ForceScheme
from sklearn.neighbors import KDTree
import random

import scipy.linalg as scp

epsilon = 1e-5


class LSP:

    def __init__(self,
                 n_neighbors=10,
                 n_components=2,
                 sample_size=0,
                 seed=7):

        self.n_neighbors_ = n_neighbors
        self.n_components_ = n_components
        self.sample_size_ = sample_size
        self.seed_ = seed
        self.embedding_ = None

    def _fit(self, X, metric):
        # define sample size
        if self.sample_size_ == 0:
            self.sample_size_ = int(len(X) / 10)

        # get a random sample
        random.seed(self.seed_)
        self.sample_index_ = random.sample(range(len(X)), self.sample_size_)
        self.X_sample_ = X[self.sample_index_, :]

        # project sample using ForceScheme
        self.y_sample_ = ForceScheme(max_it=100,
                                     n_components=self.n_components_
                                     ).fit_transform(self.X_sample_, metric=metric)

        return self

    def transform(self, X, metric='euclidean'):
        size = len(X)

        # creating the final embedding
        self.embedding_ = np.zeros((size, self.n_components_))

        # create the Laplacian matrix part of A
        tree = KDTree(X, leaf_size=2, metric=metric)
        dists, indexes = tree.query(X, k=self.n_neighbors_ + 1)

        A = np.zeros(((size + self.sample_size_), size))
        for i in range(size):
            for j in range(self.n_neighbors_):
                if indexes[i][j] != i:
                    A[i][indexes[i][j]] = 1.0
                    A[indexes[i][j]][i] = 1.0

        for i in range(size):
            A[i] = A[i] * -1.0/np.sum(A[i])
            A[i][i] = 1.0

        # add the control points to A
        for i in range(self.sample_size_):
            A[i+size][self.sample_index_[i]] = 1.0

        # create matrix b
        b = np.zeros(((size + self.sample_size_), self.n_components_))
        for i in range(self.sample_size_):
            b[i+size] = self.y_sample_[i]

        # solving Ax = b in least square sense
        self.embedding_ = scp.cho_solve(scp.cho_factor(np.dot(A.T, A)), np.dot(A.T, b))

        # adding the center of the sample projection back
        return self.embedding_

    def fit_transform(self, X, metric='euclidean'):
        self._fit(X, metric)
        return self.transform(X, metric)

    def fit(self, X, metric='euclidean'):
        return self._fit(X, metric)
