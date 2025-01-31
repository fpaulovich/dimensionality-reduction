# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT


import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from fast_force_scheme import FastForceScheme


def file_data():  # 0:03:55.414676
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
    y = FastForceScheme(max_it=100).fit_transform(X, metric='euclidean')
    end = timer()

    print('Fast ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

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
    y = FastForceScheme(max_it=100).fit_transform(X, metric='euclidean')
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
