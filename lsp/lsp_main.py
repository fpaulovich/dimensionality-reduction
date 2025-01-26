# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

# If you use this implementation, please cite
# Paulovich FV, Nonato LG, Minghim R, Levkowitz H. Least square projection: a fast high-precision multidimensional
# projection technique and its application to document mapping. IEEE Trans Vis Comput Graph. 2008 May-Jun;14(3):564-75.
# doi: 10.1109/TVCG.2007.70443.

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
import pandas as pd
import numpy as np
import random

from lsp import LSP
from sklearn.manifold import TSNE


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

    start = timer()
    lsp = LSP(n_neighbors=10,
              sample_size=int(len(X) * 0.2)).fit(X=X)
    y = lsp.transform(X=X)
    end = timer()

    print('LSP took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=label,
                cmap='viridis', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()


def main_no_sample_projection():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    start = timer()
    lsp = LSP(n_neighbors=10,
              sample_size=int(len(X) * 0.2)).fit(X=X)
    y = lsp.transform(X=X)
    end = timer()

    print('LSP took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


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
                    perplexity=10,
                    random_state=0).fit_transform(X_sample)

    lsp = LSP(n_neighbors=10).fit(X=X,
                                  X_sample=X_sample,
                                  y_sample=y_sample)
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
    main_sample_projection()
    exit(0)
