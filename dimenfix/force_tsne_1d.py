# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

from force.force_scheme import ForceScheme
from sklearn.manifold import TSNE

import sklearn.datasets as datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt

import numpy as np

DATA_TYPES = {'nominal', 'ordinal'}


def tsne_1d_fixed_feature(X, fixed_feature, feature_type='ordinal', perplexity=5):
    if feature_type not in DATA_TYPES:
        raise ValueError("results: feature_type must be one of %r." % DATA_TYPES)

    final_projection = np.zeros((len(fixed_feature), 2), dtype=np.float32)

    y = TSNE(n_components=1,
             init='random',
             perplexity=perplexity).fit_transform(X)

    # x-coordinate is the fixed feature
    if feature_type == 'nominal':
        unique_vals = list(set(fixed_feature))

        for i in range(len(fixed_feature)):
            final_projection[i][0] = (1 / len(unique_vals)) * (int(fixed_feature[i]) + 0.5)
    elif feature_type == 'ordinal':
        for i in range(len(final_projection)):
            final_projection[i][0] = fixed_feature[i]

    # y-coordinate is the t-SNE normalized
    min_y = min(y)
    max_y = max(y)

    for i in range(len(final_projection)):
        final_projection[i][1] = (y[i] - min_y) / (max_y - min_y)

    return final_projection


def force_1d_fixed_feature(X, fixed_feature, feature_type='ordinal'):
    if feature_type not in DATA_TYPES:
        raise ValueError("results: feature_type must be one of %r." % DATA_TYPES)

    final_projection = np.zeros((len(fixed_feature), 2), dtype=np.float32)

    # execute force scheme for 1D, this is the y-coordinate
    y = ForceScheme(n_components=1,
                    max_it=1000,
                    seed=None).fit_transform(X)

    # x-coordinate is the fixed feature
    if feature_type == 'nominal':
        min_y = min(y)
        max_y = max(y)
        unique_vals = list(set(fixed_feature))

        for i in range(len(fixed_feature)):
            final_projection[i][0] = (1 / len(unique_vals)) * (int(fixed_feature[i]) + 0.5)
            final_projection[i][1] = (y[i] - min_y) / (max_y - min_y)
    elif feature_type == 'ordinal':
        for i in range(len(final_projection)):
            final_projection[i][0] = fixed_feature[i]
            final_projection[i][1] = y[i]

    return final_projection


def main():
    raw = datasets.load_wine(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    label = np.array(raw.target).reshape(-1, 1)[:, 0]
    # feature = X[:, 2]

    y = force_1d_fixed_feature(X,
                               fixed_feature=label,
                               feature_type='nominal')

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=raw.target,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()


if __name__ == "__main__":
    main()
    exit(0)
