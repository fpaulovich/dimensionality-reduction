# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

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

import pandas as pd
import numpy as np


def file_data():
    input_file = "../data/cbr-ilp-ir.csv"
    df = pd.read_csv(input_file, header=0, sep='[;,]', engine='python')

    label = df[df.columns[len(df.columns) - 1]]  # get the last column as labels
    df = df.drop(labels='label', axis=1)  # removing the column class
    df = df.drop(labels='id', axis=1)  # removing the id class

    X = preprocessing.StandardScaler().fit_transform(df.values)

    # define sample size
    sample_size = int(len(X) * 0.2)

    # get a random sample
    random.seed(7)
    X_sample = X[random.sample(range(len(X)), sample_size), :]

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
    plt.scatter(y[:, 0], y[:, 1], c=label,
                cmap='Dark2', edgecolors='face', linewidths=0.25, s=5)
    plt.grid(linestyle='dotted')
    plt.show()


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
    file_data()
    exit(0)
