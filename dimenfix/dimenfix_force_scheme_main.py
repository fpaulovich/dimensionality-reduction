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
from dimenfix_force_scheme import DimenFixForceScheme

from dimenfix import rotate, split_groups, calculate_centroids, compute_positions_nominal
from sklearn.datasets import make_blobs

import pandas as pd

from sklearn.manifold import TSNE
# from t_sne_dimenfix_old import TSNEDimenfix
from force.force_scheme import ForceScheme
from t_sne_dimenfix import DimenFixTSNE


def test_centroids():
    embedding, label = make_blobs(n_samples=100, centers=10, n_features=2)

    groups = split_groups(label)
    centroids = calculate_centroids(embedding, groups)

    print(centroids)

    compute_positions_nominal(embedding, groups)

    # plt.figure()
    # plt.scatter(embedding[:, 0], embedding[:, 1], c=label,
    #             cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    # plt.grid(linestyle='dotted')
    # plt.show()


def test_groups():
    raw = datasets.load_digits(as_frame=True)
    label = raw.target

    groups = split_groups(label)

    print(len(groups))
    print(groups)


def test_rotation():
    y, labels = make_blobs(n_samples=1000, centers=3, n_features=2)

    for i in range(len(y)):
        tmp = y[i][0]
        y[i][0] = y[i][1]
        y[i][1] = tmp

    new_y = rotate(y, labels)

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=labels,
                cmap='tab10', edgecolors='face', alpha=0.1, linewidths=0.5, s=12)

    plt.scatter(new_y[:, 1], new_y[:, 0], c=labels,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    plt.grid(linestyle='dotted')
    plt.show()


def main():
    raw = datasets.load_breast_cancer(as_frame=True)

    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    label = np.array(raw.target).reshape(-1, 1)
    label = preprocessing.MinMaxScaler().fit_transform(label)

    fixed_feature = label[:, 0]
    # fixed_feature = X[:, 2]

    start = timer()
    # y = DimenFixForceScheme(max_it=500,
    #                         iterations_to_pull=10,
    #                         fixed_feature=fixed_feature,
    #                         feature_type='nominal',
    #                         pulling_strategy='gaussian',
    #                         alpha=1.0).fit_transform(X)
    y = DimenFixTSNE(n_components=2,
                     init='random',
                     perplexity=20,
                     iterations_to_pull=10,
                     fixed_feature=fixed_feature,
                     feature_type='nominal',
                     pulling_strategy='clipping',
                     alpha=1.0).fit_transform(X)
    end = timer()

    print('ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=fixed_feature,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    plt.grid(linestyle='dotted')
    plt.colorbar()
    plt.show()

    return


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

    # label_dist = np.zeros(len(X))
    # target = 0
    # for i in range(len(X)):
    #     label_dist[i] = math.dist(X[target], X[i])
    # label_dist = 1 - preprocessing.MinMaxScaler().fit_transform(label_dist.reshape(-1, 1))[:, 0]

    # print(label_dist)

    start = timer()
    y = DimenFixTSNE(n_components=2,
                     init='random',
                     perplexity=10,
                     iterations_to_pull=1,
                     fixed_feature=label,
                     feature_type='ordinal',
                     pulling_strategy='gaussian',
                     alpha=1.0
                     ).fit_transform(X)

    #
    # # range_limits = np.column_stack((label, label))
    # # y = TSNEDimenfix(perplexity=10,
    # #                  init='random',
    # #                  # method='exact',
    # #                  dimenfix=True,
    # #                  range_limits=range_limits,
    # #                  alpha=1.0,
    # #                  density_adj=True,
    # #                  class_ordering='disable',
    # #                  rotation=False,
    # #                  class_label=label_dist,
    # #                  fix_iter=5,
    # #                  mode='gaussian',
    # #                  early_push=False,
    # #                  random_state=42).fit_transform(X)
    # end = timer()
    # print('t-SNE took {0} to execute'.format(timedelta(seconds=end - start)))

    # start = timer()
    # y = ForceScheme(max_it=100).fit_transform(X)

    # y = DimenFixForceScheme(max_it=100,
    #                         iterations_to_pull=1,
    #                         fixed_feature=label,
    #                         feature_type='ordinal',
    #                         pulling_strategy='gaussian',
    #                         alpha=1.0).fit_transform(X)
    end = timer()
    print('ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=label, cmap='viridis', edgecolors='face', linewidths=0.5, s=12)
    # plt.scatter(y[target][1], y[target][0], c='red', edgecolors='face', linewidths=0.5, s=20)

    # plt.grid(linestyle='dotted')
    plt.colorbar(label='Similarity')
    plt.show()


if __name__ == "__main__":
    main()
    exit(0)
