import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import math

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from force.pbc import PBC

from sklearn.cluster import KMeans


def main():
    raw = datasets.load_breast_cancer(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)

    kmeans = KMeans(init='random',
                    algorithm='lloyd',
                    random_state=None,
                    n_clusters=3).fit(X)
    labels = kmeans.labels_

    start = timer()
    y = PBC(labels=labels).fit_transform(X)
    end = timer()

    print(np.amin(y, axis=0))

    print('ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    main()
    exit(0)
