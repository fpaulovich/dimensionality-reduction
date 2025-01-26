# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

# If you use this implementation, please cite
# Paulovich FV, Silva CT, Nonato LG. Two-phase mapping for projecting massive data sets. IEEE Trans Vis Comput Graph.
# 2010 Nov-Dec;16(6):1281-90. doi: 10.1109/TVCG.2010.207.

import numpy as np
from force.force_scheme import ForceScheme

epsilon = 1e-5


class PLMP:

    def __init__(self,
                 n_components=2):
        self.n_components_ = n_components
        self.sample_size_ = 0
        self.X_sample_ = None
        self.y_sample_ = None
        self.embedding_ = None

    def _fit(self, X_sample, y_sample, metric):
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

        # sample data columns cannot be null
        self.X_sample_ = self.X_sample_ + epsilon

        # centering sample data and projection
        self.y_sample_center_ = np.average(self.y_sample_, axis=0)
        self.y_sample_ = np.subtract(self.y_sample_, self.y_sample_center_)

        self.X_sample_center_ = np.average(self.X_sample_, axis=0)
        self.X_sample_ = np.subtract(self.X_sample_, self.X_sample_center_)

        # making sample data non-singular
        for i in range(len(self.X_sample_[0])):
            self.X_sample_[i][i] = 1.0

        # finding A where(D'.A = P') in a least-square sense
        D = self.X_sample_
        P = self.y_sample_
        self.A = np.linalg.solve(np.dot(D.T, D), np.dot(D.T, P))

        return self

    def transform(self, X):
        self.embedding_ = np.zeros((len(X), self.n_components_))

        # centering the data using the sample data center
        X = np.subtract(X, self.X_sample_center_)

        # project the data using A (P = X * A)
        self.embedding_ = np.dot(X, self.A)

        # adding the center of the sample projection back
        return np.add(self.embedding_, self.y_sample_center_)

    def fit_transform(self, X, X_sample, y_sample=None, metric='euclidean'):
        self._fit(X_sample, y_sample, metric)
        return self.transform(X)

    def fit(self, X_sample, y_sample=None, metric='euclidean'):
        return self._fit(X_sample, y_sample, metric)
