# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright 2024 Fernando V. Paulovich
# License: BSD-3-Clause
# code inspired by https://github.com/lgnonato/LAMP/blob/master/lamp.py

# If you use this implementation, please cite
# P. Joia, D. Coimbra, J. A. Cuminato, F. V. Paulovich and L. G. Nonato, "Local Affine Multidimensional Projection,"
# in IEEE Transactions on Visualization and Computer Graphics, vol. 17, no. 12, pp. 2563-2571, Dec. 2011,
# doi: 10.1109/TVCG.2011.220.

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing

from lamp import Lamp
from sklearn.manifold import TSNE
import random
import math


def main_sample_projection():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    # define sample size
    sample_size = int(len(X) * 0.2)

    # get a random sample
    random.seed(7)
    X_sample = X[random.sample(range(len(X)), sample_size), :]

    # project the sample
    start = timer()
    y_sample = TSNE(n_components=2,
                    perplexity=12,
                    random_state=0).fit_transform(X_sample)

    lamp = Lamp(nr_neighbors=12).fit(X_sample=X_sample,
                                     y_sample=y_sample)
    y = lamp.transform(X=X)
    end = timer()

    print('Lamp took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y_sample[:, 0], y_sample[:, 1], c='white',
                edgecolors='black', linewidths=0.5, s=20)

    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


def main_no_sample_projection():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    # define sample size
    sample_size = int(math.sqrt(len(X)))
    sample_size = max(sample_size, 2 * len(X[0]))

    # get a random sample
    random.seed(7)
    X_sample = X[random.sample(range(len(X)), sample_size), :]

    start = timer()
    lamp = Lamp(nr_neighbors=10).fit(X_sample=X_sample)
    y = lamp.transform(X=X)
    end = timer()

    print('Lamp took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    main_sample_projection()
    exit(0)
