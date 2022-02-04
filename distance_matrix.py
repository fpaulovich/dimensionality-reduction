import math
import numpy as np

from scipy.spatial import distance
from sklearn.datasets import load_iris


class DistanceMatrix:

    def __init__(self):
        self.size_: int = 0
        self.distance_matrix_ = None
        self.total_: int = 0

    def _fit(self, X, distance_function):
        self.size_ = len(X)
        self.distance_matrix_ = np.zeros(int(self.size_ * (self.size_ + 1) / 2), dtype=np.dtype('f8'))
        self.total_ = len(self.distance_matrix_)

        k = 0
        for i in range(self.size_):
            for j in range(i, self.size_):
                self.distance_matrix_[k] = distance_function(X[i], X[j])
                k = k + 1

        return self

    def fit_transform(self, X, distance_function=distance.euclidean):
        self._fit(X, distance_function)
        return self

    def fit(self, X, distance_function=distance.euclidean):
        self._fit(X, distance_function)
        return self

    def size(self) -> int:
        return self.size_

    def get(self, i: int, j: int) -> float:
        r = (i + j - math.fabs(i - j)) / 2  # min(i,j)
        s = (i + j + math.fabs(i - j)) / 2  # max(i,j)
        return self.distance_matrix_[int(self.total_ - ((self.size_ - r) * (self.size_ - r + 1) / 2) + (s - r))]


def main():
    # exemplos de funcao de distancia
    # https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

    X = load_iris(as_frame=True).data.to_numpy()
    Y = DistanceMatrix().fit_transform(X, distance.euclidean)
    print(Y.distance_matrix_)

    return


if __name__ == "__main__":
    main()
    exit(0)
