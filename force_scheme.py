import numpy as np
from numpy import random
import math
from numba import jit
from distance_matrix import DistanceMatrix

import distance_matrix as dm
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta


class ForceScheme:

    def __init__(self):
        self.max_it_ = 100
        self.learning_rate0_ = 0.5
        self.decay_ = 0.95

        # create the initial projection
        self.projection_ = None

        # pca = decomposition.PCA(n_components=2).fit(X)
        # Y = pca.transform(X)

    def execute(self, dataset, distance_function):
        # create distance matrix
        distance_matrix = dm.DistanceMatrix(dataset, distance_function)
        return self.execute(distance_matrix)

    def execute(self, distance_matrix: DistanceMatrix):
        # create the initial projection
        self.projection_ = np.random.random((distance_matrix.size(), 2))

        for k in range(self.max_it_):
            start = timer()

            learning_rate = self.learning_rate0_ * math.pow((1 - k / self.max_it_), self.decay_)
            error = self.iteration(learning_rate, distance_matrix)

            end = timer()
            print('Force iteration #{0}-{1} time: {2} error:{3}'.format((k + 1),
                                                                        self.max_it_,
                                                                        timedelta(seconds=end - start),
                                                                        error))

        # setting the min to (0,0)
        min_x = min(self.projection_[:, 0])
        min_y = min(self.projection_[:, 1])
        for i in range(len(self.projection_)):
            self.projection_[i][0] -= min_x
            self.projection_[i][1] -= min_y

        return self.projection_

    @jit
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

    @jit
    def move(self, pivot, learning_rate: float, distance_matrix: DistanceMatrix):
        error = 0.0
        size = distance_matrix.size()

        for ins in range(size):

            if pivot != ins:
                x1x2 = self.projection_[ins][0] - self.projection_[pivot][0]
                y1y2 = self.projection_[ins][1] - self.projection_[pivot][1]
                dr2 = math.sqrt(x1x2 * x1x2 + y1y2 * y1y2)

                if dr2 < 0.0001:
                    dr2 = 0.0001

                # getting te index in the distance matrix and getting the value
                drn = distance_matrix.get(pivot, ins)

                # calculate the movement
                delta = (drn - dr2) * math.fabs(drn - dr2)
                error += math.fabs(delta)

                # moving
                self.projection_[ins][0] = self.projection_[ins][0] + learning_rate * delta * (x1x2 / dr2)
                self.projection_[ins][1] = self.projection_[ins][1] + learning_rate * delta * (y1y2 / dr2)

        return error / size


def main():
    # exemplos de funcao de distancia
    # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    # # input_file = "/Users/fpaulovich/Dropbox/datasets/data/19k_fourier30_spatial10_normalized.csv"
    # # input_file = "/Users/fpaulovich/Dropbox/datasets/data/segmentation-normcols.csv"
    # input_file = "/Users/fpaulovich/Dropbox/datasets/data/mammals-50000.bin-normcols.csv"
    #
    # dataset = pd.read_csv(input_file)
    # labels = dataset.loc[:, 'label']
    # dataset = dataset.drop(['id', 'label'], axis=1) #remove ids and labels
    #
    # print(dataset)
    #
    # force = ForceScheme(dataset)
    # force.max_it = 10
    # projection = force.execute(distance.euclidean)
    #
    # plt.figure()
    # plt.scatter(projection[:, 0], projection[:, 1], c=labels,
    #             cmap='Set2', edgecolors='black', linewidths=0.5)
    # plt.grid(linestyle='dotted')
    # plt.show()

    ####################

    # input_file = "/Users/fpaulovich/Dropbox/datasets/data/segmentation-normcols.csv"
    # dataset = pd.read_csv(input_file)
    # labels = dataset.loc[:, 'label']
    #
    # input_file = "/Users/fpaulovich/Desktop/segmentation_distance_matrix.txt"
    # dmat = dm.DistanceMatrix().read(input_file)
    #
    # force = ForceScheme()
    # force.max_it_ = 5
    # projection = force.execute(dmat)
    #
    # plt.figure()
    # plt.scatter(projection[:, 0], projection[:, 1], c=labels,
    #             cmap='tab10', edgecolors='face', linewidths=0.5, s=4)
    # plt.grid(linestyle='dotted')
    # plt.show()

    #####################

    start = timer()

    # input_file = "/Users/fpaulovich/Desktop/matriz_distancia"
    input_file = "/Users/fpaulovich/Documents/protein_folding/dmat_8000"
    dmat = dm.DistanceMatrix().read(input_file)

    force = ForceScheme()
    force.max_it_ = 10
    projection = force.execute(dmat)

    end = timer()
    print('Total time: ', timedelta(seconds=end - start))

    labels = np.zeros(len(projection))

    plt.figure()
    plt.scatter(projection[:, 0], projection[:, 1], c=labels,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    ####################

    return


if __name__ == "__main__":
    main()
    exit(0)
