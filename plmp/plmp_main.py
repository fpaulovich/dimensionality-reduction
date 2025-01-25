# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

# If you use this implementation, please cite
# Paulovich FV, Silva CT, Nonato LG. Two-phase mapping for projecting massive data sets. IEEE Trans Vis Comput Graph.
# 2010 Nov-Dec;16(6):1281-90. doi: 10.1109/TVCG.2010.207.

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing

from sklearn.manifold import TSNE
from plmp import PLMP
import random
import math

import pandas as pd
import numpy as np


def song_data():
    data_file = "../data/song_data.csv"
    df = pd.read_csv(data_file, header=0, engine='python')

    df = df.sort_values(by='song_popularity', ascending=True)
    df = df.drop_duplicates('song_name', keep='last')  # drop duplicates, keep the largest song_popularity

    label = np.array(df['song_popularity'].values).reshape(-1, 1)
    label = preprocessing.MinMaxScaler().fit_transform(label)[:, 0]

    df = df.drop(['song_name', 'song_popularity'], axis=1)
    X = df.values
    X = preprocessing.MinMaxScaler().fit_transform(X)

    # define sample size
    sample_size = int(len(X) * 0.2)

    # get a random sample
    random.seed(7)
    X_sample = X[random.sample(range(len(X)), sample_size), :]

    start = timer()
    y_sample = TSNE(n_components=2,
                    perplexity=12,
                    random_state=0).fit_transform(X_sample)

    plmp = PLMP().fit(X_sample=X_sample,
                      y_sample=y_sample)
    y = plmp.transform(X=X)
    end = timer()

    print('PLMP took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=label,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
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

    plmp = PLMP().fit(X_sample=X_sample,
                      y_sample=y_sample)
    y = plmp.transform(X=X)
    end = timer()

    print('PLMP took {0} to execute'.format(timedelta(seconds=end - start)))

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
    plmp = PLMP().fit(X_sample=X_sample)
    y = plmp.transform(X=X)
    end = timer()

    print('PLMP took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    song_data()
    exit(0)
