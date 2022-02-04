import math
import pandas as pd
import numpy as np
from numba import jit

from scipy.spatial import distance

from timeit import default_timer as timer
from datetime import timedelta
from sys import getsizeof


class DistanceMatrix:

    def __init__(self):
        self.size_: int = 0
        self.distance_matrix_ = None
        self.total_: int = 0

    @jit
    def create(self, dataset, distance_function):
        start = timer()

        self.size_ = len(dataset)
        self.distance_matrix_ = np.zeros(int(self.size_ * (self.size_ + 1) / 2), dtype=np.dtype('f8'))
        self.total_ = len(self.distance_matrix_)
        data = dataset.to_numpy()

        print('Distance matrix size in memory: ', round(getsizeof(self.distance_matrix_) / 1024 / 1024, 2), 'MB')

        k = 0
        for i in range(self.size_):
            for j in range(i, self.size_):
                self.distance_matrix_[k] = distance_function(data[i], data[j])
                k = k + 1

        end = timer()
        print('Time to create the distance matrix: ', timedelta(seconds=end - start))

        return self

    def read(self, filename):
        start = timer()

        self.size_ = 0

        try:
            # calculating the size
            fp = open(filename)
            line = fp.readline()
            tokens = line.strip().split(' ')
            self.size_ = len(tokens)

            self.distance_matrix_ = np.zeros(int(self.size_ * (self.size_ + 1) / 2), dtype=np.dtype('f8'))
            self.total_ = len(self.distance_matrix_)

            # reading line per line
            row = 0
            k = 0
            with open(filename) as fp:
                line = fp.readline()

                while line:
                    tokens = line.strip().split(' ')

                    for column in range(row, self.size_):
                        self.distance_matrix_[k] = float(tokens[column])
                        k = k + 1

                    line = fp.readline()
                    row = row + 1

        finally:
            fp.close()

        print('Distance matrix size in memory: ', round(getsizeof(self.distance_matrix_) / 1024 / 1024, 2), 'MB')

        end = timer()
        print('Time to read the distance matrix: ', timedelta(seconds=end - start))

        return self

    @jit
    def size(self) -> int:
        return self.size_

    @jit
    def get(self, i: int, j: int) -> float:
        r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
        s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
        return self.distance_matrix_[int(self.total_ - ((self.size_ - r) * (self.size_ - r + 1) / 2) + (s - r))]


def main():
# exemplos de funcao de distancia
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    # input_file = "/Users/fpaulovich/Dropbox/datasets/data/19k_fourier30_spatial10_normalized.csv"
    #
    # dataset = pd.read_csv(input_file)
    # dm = DistanceMatrix(dataset, distance.euclidean)

    # input_file = "/Users/fpaulovich/Dropbox/datasets/data/segmentation-normcols.csv"
    # dataset = pd.read_csv(input_file)
    # labels = dataset.loc[:, 'label']
    # dataset = dataset.drop(['id', 'label'], axis=1) #remove ids and labels
    # dmat = DistanceMatrix().create(dataset, distance.euclidean)
    #
    # data = np.repeat(np.zeros(dmat.size()), dmat.size())
    # data = data.reshape(dmat.size(), dmat.size())
    #
    # for i in range(dmat.size()):
    #     for j in range(dmat.size()):
    #         data[i][j] = dmat.get(i, j)
    #
    # np.savetxt("/Users/fpaulovich/Desktop/segmentation_distance_matrix.txt", data, delimiter=" ", fmt="%s")

    input_file = "/Users/fpaulovich/Desktop/matriz_distancia"
    dmat = DistanceMatrix().read(input_file)

    begin = 0
    end = 8000
    size = end-begin
    data = np.repeat(np.zeros(size), size)
    data = data.reshape(size, size)

    for i in range(size):
        for j in range(size):
            data[i][j] = dmat.get(i+begin, j+begin)

    np.savetxt("/Users/fpaulovich/Desktop/AB40_distance_matrix.txt", data, delimiter=" ", fmt="%s")

    return


if __name__ == "__main__":
    main()
    exit(0)
