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
import random

from lsp import LSP
from sklearn.manifold import TSNE


def file_data():
    input_file = "../data/cbr-ilp-ir.csv"
    df = pd.read_csv(input_file, header=0, sep='[;,]', engine='python')

    label = df[df.columns[len(df.columns) - 1]]  # get the last column as labels
    df = df.drop(labels='label', axis=1)  # removing the column class
    df = df.drop(labels='id', axis=1)  # removing the id class

    X = preprocessing.StandardScaler().fit_transform(df.values)

    start = timer()
    lsp = LSP(n_neighbors=10,
              sample_size=int(len(X) * 0.2)).fit(X=X)
    y = lsp.transform(X=X)
    end = timer()

    print('LSP took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=label,
                cmap='Dark2', edgecolors='black', linewidths=0.25, s=10)
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
    plt.scatter(y_sample[:, 0], y_sample[:, 1], c='white',
                edgecolors='black', linewidths=0.5, s=20)

    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    file_data()
    exit(0)
