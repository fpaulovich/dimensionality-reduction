# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright 2024 Fernando V. Paulovich
# License: BSD-3-Clause
# code inspired by https://github.com/lgnonato/LAMP/blob/master/lamp.py

# If you use this implementation, please cite
# Paulovich FV, Silva CT, Nonato LG. Two-phase mapping for projecting massive data sets. IEEE Trans Vis Comput Graph.
# 2010 Nov-Dec;16(6):1281-90. doi: 10.1109/TVCG.2010.207.

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing

from sklearn.manifold import TSNE
from lsp import LSP
import random
import math


# def main_sample_projection():
#     raw = datasets.load_digits(as_frame=True)
#     X = raw.data.to_numpy()
#     X = preprocessing.MinMaxScaler().fit_transform(X)
#
#     # define sample size
#     sample_size = int(len(X) * 0.2)
#
#     # get a random sample
#     random.seed(7)
#     X_sample = X[random.sample(range(len(X)), sample_size), :]
#
#     # project the sample
#     y_sample = TSNE(n_components=2,
#                     perplexity=12,
#                     random_state=0).fit_transform(X_sample)
#
#     start = timer()
#     plmp = PLMP().fit(X_sample=X_sample,
#                       y_sample=y_sample)
#     y = plmp.transform(X=X)
#     end = timer()
#
#     print('PLMP took {0} to execute'.format(timedelta(seconds=end - start)))
#
#     plt.figure()
#     plt.scatter(y_sample[:, 0], y_sample[:, 1], c='white',
#                 edgecolors='black', linewidths=0.5, s=20)
#
#     plt.scatter(y[:, 0], y[:, 1], c=raw.target,
#                 cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
#     plt.grid(linestyle='dotted')
#     plt.show()
#
#     return


def main_no_sample_projection():
    raw = datasets.load_breast_cancer(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    start = timer()
    lsp = LSP(n_neighbors=20).fit(X=X)
    y = lsp.transform(X=X)
    end = timer()

    print('LSP took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    main_no_sample_projection()
    exit(0)
