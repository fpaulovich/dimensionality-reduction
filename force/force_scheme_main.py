# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

# If you use this implementation, please cite
# Rosane Minghim, Fernando Vieira Paulovich, and Alneu de Andrade Lopes "Content-based text mapping using
# multi-dimensional projections for exploration of document collections", Proc. SPIE 6060, Visualization and Data
# Analysis 2006, 60600S (16 January 2006); https://doi.org/10.1117/12.650880

import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from force.force_scheme import ForceScheme


def file_data():
    # ForceScheme took 0:01:50.905411 to execute
    # Number iterations: 99
    # Error: 0.22554391757561443
    # ForceScheme took 0:05:47.876620 to execute

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
    y = ForceScheme(max_it=100).fit_transform(X, metric='euclidean')
    end = timer()

    print('ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=label,
                cmap='Dark2', edgecolors='black', linewidths=0.25, s=10)
    plt.grid(linestyle='dotted')
    plt.show()


def main():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)

    start = timer()
    y = ForceScheme(max_it=100).fit_transform(X, metric='euclidean')
    end = timer()

    print('ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    file_data()
    exit(0)
