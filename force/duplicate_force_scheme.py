from sklearn.neighbors import KDTree

import sklearn.datasets as datasets
from sklearn import preprocessing

import numpy as np
import skdim

import matplotlib.pyplot as plt

from force.force_scheme import ForceScheme
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy import stats


def local_dimensionality(X, k):
    # estimate global intrinsic dimension
    glob = skdim.id.lPCA().fit_transform(X)

    # estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
    loc = skdim.id.lPCA().fit_transform_pw(X,
                                           n_neighbors=k,
                                           n_jobs=1)

    return glob, loc


def point_stress(X, y):
    stress = np.zeros(len(X))

    for i in range(len(X)):
        for j in range(len(X)):
            drn = np.sqrt(np.sum(np.square(X[i] - X[j])))
            dr2 = np.sqrt(np.sum(np.square(y[i] - y[j])))
            stress[i] += (drn - dr2) * (drn - dr2)
    return np.sqrt(stress)


# def influency_stress(X, y, M):
#     stress = np.zeros(len(X))
#
#     for i in range(len(X)):
#         for j in range(len(X)):
#             #


def symmetrize_adjacency_nearest_neighbor_matrix(M):
    SM = np.zeros((len(M), len(M)))

    for i in range(len(M)):
        for j in range(len(M)):
            SM[i][j] = max(M[i][j], M[j][i])

    return SM


def adjacency_nearest_neighbor_matrix(X, k):
    tree = KDTree(X, leaf_size=30, metric='euclidean')
    knn = tree.query(X, k=k, return_distance=False)

    matrix = np.zeros((len(X), len(X)))

    for i in range(len(knn)):
        for j in range(k):
            matrix[i][knn[i][j]] = 1

    return matrix


def count_neighbors(matrix):
    count = np.zeros(len(matrix))

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j] == 1:
                count[j] += 1

    return count


def main():
    raw = datasets.load_breast_cancer(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)

    # k = 15  # number of neighbors
    #
    # adj_matrix = symmetrize_adjacency_nearest_neighbor_matrix(adjacency_nearest_neighbor_matrix(X, k))
    # count = count_neighbors(adj_matrix)
    #
    # # plt.figure()
    # # plt.bar(np.arange(len(count)), np.sort(count), align='center', alpha=0.5)
    # # plt.show()
    #
    # # y = ForceScheme().fit_transform(X)
    # y = PCA(n_components=2).fit_transform(X)
    # # y = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
    #
    # stress = point_stress(X, y)
    #
    # # plt.figure()
    # # plt.bar(np.arange(len(stress)), np.sort(stress), align='center', alpha=0.5)
    # # plt.show()
    #
    # print(stats.pearsonr(count, stress))
    #
    # plt.scatter(count, stress, c=count, cmap='plasma', edgecolors='face', linewidths=0.5, s=20)
    # # plt.scatter(y[:, 0], y[:, 1], c=count, cmap='plasma', edgecolors='face', linewidths=0.5, s=20)
    # m, b = np.polyfit(count, stress, 1)
    # plt.plot(count, m * count + b, color='red')
    # plt.grid(linestyle='dotted')
    # plt.colorbar(label="# neighbors", orientation="horizontal")
    # plt.show()

    k = int(len(X) / 10)  # number of neighbors
    global_dim, local_dim = local_dimensionality(X, k)

    # plt.figure()
    # plt.bar(np.arange(len(local_dim)), np.sort(local_dim), align='center', alpha=0.5)
    # plt.plot(np.arange(len(local_dim)), [global_dim] * len(local_dim))
    # plt.show()

    y = ForceScheme().fit_transform(X)
    # y = PCA(n_components=2).fit_transform(X)
    # y = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

    stress = point_stress(X, y)

    plt.scatter(local_dim, stress, c=local_dim, cmap='plasma', edgecolors='face', linewidths=0.5, s=20)
    m, b = np.polyfit(local_dim, stress, 1)
    plt.plot(local_dim, m * local_dim + b, color='red')
    plt.grid(linestyle='dotted')
    plt.colorbar(label="intrinsic dimensionality", orientation="horizontal")
    plt.show()

    print(stats.pearsonr(local_dim, stress))


if __name__ == "__main__":
    main()
    exit(0)
