import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from dimenfix_force_scheme import DimenFixForceScheme

from dimenfix import rotate, split_groups, calculate_centroids, compute_positions_nominal
from sklearn.datasets import make_blobs


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
    # fixed_feature = X[:, 5]

    start = timer()
    y = DimenFixForceScheme(max_it=100,
                            iterations_to_pull=5,
                            fixed_feature=fixed_feature,
                            feature_type='nominal',
                            pulling_strategy='rescale',
                            alpha=1.0).fit_transform(X)
    end = timer()

    # print(np.amin(y, axis=0))

    print('ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=fixed_feature,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=12)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    main()
    # test_rotation()
    exit(0)
