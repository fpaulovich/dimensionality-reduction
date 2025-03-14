# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
import pandas as pd

import math

from dimenfix_t_sne import DimenFixTSNE


def main():
    raw = datasets.load_digits(as_frame=True)

    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    label = np.array(raw.target).reshape(-1, 1)
    # label = preprocessing.MinMaxScaler().fit_transform(label)

    fixed_feature = label[:, 0]
    # fixed_feature = X[:, 2]

    start = timer()
    y = DimenFixTSNE(n_components=2,
                     init='random',
                     perplexity=15,
                     iterations_to_pull=10,
                     fixed_feature=fixed_feature,
                     feature_type='nominal',
                     pulling_strategy='gaussian',
                     alpha=1.0).fit_transform(X)
    end = timer()

    print('DimenFixTSNE took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=fixed_feature,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    plt.grid(linestyle='dotted')
    plt.colorbar()
    plt.show()

    return


def main_3D():
    raw = datasets.load_digits(as_frame=True)

    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    label = np.array(raw.target).reshape(-1, 1)
    # label = preprocessing.MinMaxScaler().fit_transform(label)

    fixed_feature = label[:, 0]
    # fixed_feature = X[:, 2]

    start = timer()
    y = DimenFixTSNE(n_components=3,
                     init='random',
                     perplexity=15,
                     iterations_to_pull=10,
                     fixed_feature=fixed_feature,
                     feature_type='nominal',
                     pulling_strategy='rescale',
                     alpha=1.0).fit_transform(X)
    end = timer()

    print('DimenFixTSNE took {0} to execute'.format(timedelta(seconds=end - start)))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


def song_data():
    data_file = "../data/song_data.csv"
    df = pd.read_csv(data_file, header=0, engine='python')

    df = df.sort_values(by='song_popularity', ascending=True)
    df = df.drop_duplicates('song_name', keep='last')  # drop duplicates, keep the largest song_popularity

    label = np.array(df['song_popularity'].values).reshape(-1, 1)

    df = df.drop(['song_name', 'song_popularity'], axis=1)
    X = df.values
    X = preprocessing.MinMaxScaler().fit_transform(X)

    # create a label based on the distance to a song
    label_dist = np.zeros(len(X))
    target = 0
    for i in range(len(X)):
        label_dist[i] = math.dist(X[target], X[i])
    label_dist = 1 - preprocessing.MinMaxScaler().fit_transform(label_dist.reshape(-1, 1))[:, 0]

    start = timer()
    y = DimenFixTSNE(n_components=2,
                     init='random',
                     perplexity=30,
                     iterations_to_pull=10,
                     fixed_feature=label_dist,
                     feature_type='ordinal',
                     pulling_strategy='gaussian',
                     alpha=1.0
                     ).fit_transform(X)
    end = timer()
    print('DimenFixTSNE took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=label_dist, cmap='viridis', edgecolors='face', linewidths=0.5, s=12)
    plt.scatter(y[target][1], y[target][0], c='red', edgecolors='face', linewidths=0.5, s=20)
    plt.colorbar(label='Similarity')
    plt.show()


if __name__ == "__main__":
    main_3D()
    exit(0)
