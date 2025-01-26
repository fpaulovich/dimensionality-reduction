# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

# If you use this implementation, please cite
# Paulovich FV, Nonato LG, Minghim R, Levkowitz H. Least square projection: a fast high-precision multidimensional
# projection technique and its application to document mapping. IEEE Trans Vis Comput Graph. 2008 May-Jun;14(3):564-75.
# doi: 10.1109/TVCG.2007.70443.

import numpy as np
from force.force_scheme import ForceScheme
from sklearn.neighbors import KDTree
import random

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

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
        self.X_sample_ = None
        self.y_sample_ = None
        self.sample_index_ = None

    def _fit(self, X, X_sample, y_sample, metric):
        # define sample size
        if self.sample_size_ == 0:
            self.sample_size_ = int(len(X) * 0.1)

        if len(X) < self.n_neighbors_:
            raise ValueError('The n_neighbors need to be small than the X.')

        # if sample is not provided, create a sample
        if X_sample is None:
            # get a random sample
            random.seed(self.seed_)
            self.sample_index_ = random.sample(range(len(X)), self.sample_size_)
            self.X_sample_ = X[self.sample_index_, :]
        else:
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

        return self

    def transform(self, X, metric='euclidean'):
        size = len(X)

        # creating the final embedding
        self.embedding_ = np.zeros((size, self.n_components_))

        # create the Laplacian part of A
        tree = KDTree(X, leaf_size=2, metric=metric)
        dists, indexes = tree.query(X, k=self.n_neighbors_ + 1)

        # create the laplacian matrix part
        A = lil_matrix(((size + self.sample_size_), size))
        for i in range(size):
            for j in range(self.n_neighbors_):
                A[(i, indexes[i][j])] = 1.0 / (dists[i][j] + epsilon)
                A[(indexes[i][j], i)] = 1.0 / (dists[i][j] + epsilon)
            A[(i, i)] = 1.0

        # if sample is provided, find the indexes for control points
        if self.sample_index_ is None:
            self.sample_index_ = np.ravel(tree.query(self.X_sample_, k=1, return_distance=False))

        # add the control points equations
        weight = 20.0  # define the strength to map a control point to its original position
        for i in range(self.sample_size_):
            A[(i + size, self.sample_index_[i])] = weight

        # normalize so the summation of each line of the Laplacian is zero
        A = csr_matrix(A)
        for i in range(size):
            A[i] = csr_matrix.dot(A[i], -1.0 / (csr_matrix.sum(A[i]) - 1.0))
            A[(i, i)] = 1.0

        # create matrix b
        b = lil_matrix(((size + self.sample_size_), self.n_components_))
        for i in range(self.sample_size_):
            b[i + size] = weight * self.y_sample_[i]
        b = csr_matrix(b)

        # solving A^t.Ax = A^t.b in least square sense
        self.embedding_ = spsolve(csr_matrix.dot(A.T, A), csr_matrix.dot(A.T, b)).toarray()

        # adding the center of the sample projection back
        return self.embedding_

    def fit_transform(self, X, X_sample=None, y_sample=None, metric='euclidean'):
        self._fit(X, X_sample, y_sample, metric)
        return self.transform(X, metric)

    def fit(self, X, X_sample=None, y_sample=None, metric='euclidean'):
        return self._fit(X, X_sample, y_sample, metric)
