# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

# If you use this implementation, please cite
# F. V. Paulovich and R. Minghim, "Text Map Explorer: a Tool to Create and Explore Document Maps," Tenth International
# Conference on Information Visualisation (IV'06), London, UK, 2006, pp. 245-251, doi: 10.1109/IV.2006.104.

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from pbc import PBC

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

    start = timer()
    y = PBC(cluster_factor=0.2).fit_transform(X)
    end = timer()

    print('ProjClus took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=label,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()


def main():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    start = timer()
    y = PBC(cluster_factor=0.2).fit_transform(X)
    end = timer()

    print('ProjClus took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    song_data()
    exit(0)
