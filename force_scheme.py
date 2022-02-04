import numpy as np
from numpy import random
import math
from sklearn import decomposition

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from distance_matrix import DistanceMatrix
from scipy.spatial import distance
import distance_matrix as dm


class ForceScheme:

    def __init__(self,
                 max_it=100,
                 learning_rate0=0.5,
                 decay=0.95):

        self.max_it_ = max_it
        self.learning_rate0_ = learning_rate0
        self.decay_ = decay
        self.embedding_ = None

    def iteration(self, learning_rate: float, distance_matrix: DistanceMatrix):
        error = 0.0
        size = distance_matrix.size()

        # create random index
        index = np.arange(size)
        np.random.shuffle(index)

        for i in range(size):
            pivot = index[i]
            error += self.move(pivot, learning_rate, distance_matrix)

        return error / size

    def move(self, pivot, learning_rate: float, distance_matrix: DistanceMatrix):
        error = 0.0
        size = distance_matrix.size()

        for ins in range(size):

            if pivot != ins:
                x1x2 = self.embedding_[ins][0] - self.embedding_[pivot][0]
                y1y2 = self.embedding_[ins][1] - self.embedding_[pivot][1]
                dr2 = math.sqrt(x1x2 * x1x2 + y1y2 * y1y2)

                if dr2 < 0.0001:
                    dr2 = 0.0001

                # getting te index in the distance matrix and getting the value
                drn = distance_matrix.get(pivot, ins)

                # calculate the movement
                delta = (drn - dr2) * math.fabs(drn - dr2)
                error += math.fabs(delta)

                # moving
                self.embedding_[ins][0] = self.embedding_[ins][0] + learning_rate * delta * (x1x2 / dr2)
                self.embedding_[ins][1] = self.embedding_[ins][1] + learning_rate * delta * (y1y2 / dr2)

        return error / size

    def _fit(self, X, y, distance_matrix):
        # set the initial projection
        if y is None:
            pca = decomposition.PCA(n_components=2).fit(X)
            self.embedding_ = pca.transform(X)
        else:
            self.embedding_ = y

        for k in range(self.max_it_):
            learning_rate = self.learning_rate0_ * math.pow((1 - k / self.max_it_), self.decay_)
            error = self.iteration(learning_rate, distance_matrix)

        # setting the min to (0,0)
        min_x = min(self.embedding_[:, 0])
        min_y = min(self.embedding_[:, 1])
        for i in range(len(self.embedding_)):
            self.embedding_[i][0] -= min_x
            self.embedding_[i][1] -= min_y

        return self.embedding_

    def fit_transform(self, X, y=None, distance_function=distance.euclidean):
        distance_matrix = dm.DistanceMatrix().fit_transform(X, distance_function)
        return self._fit(X, y, distance_matrix)

    def fit(self, X, y=None, distance_function=distance.euclidean):
        distance_matrix = dm.DistanceMatrix().fit_transform(X, distance_function)
        return self._fit(X, y, distance_matrix)


def main():

    raw = load_iris(as_frame=True)
    X = raw.data.to_numpy()
    y = ForceScheme().fit_transform(X)

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    main()
    exit(0)
