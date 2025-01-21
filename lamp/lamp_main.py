import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing

from lamp import Lamp
from sklearn.manifold import TSNE
import random
import math


def main_tsne():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)

    # get a random sample
    random.seed(7)
    sample_size = int(len(X) * 0.2)
    indexes = random.sample(range(len(X)), sample_size)
    X_sample = np.zeros((sample_size, len(X[0])))
    for i in range(sample_size):
        X_sample[i] = X[indexes[i]]

    # project the sample
    y_sample = TSNE(n_components=2,
                    perplexity=10,
                    random_state=0).fit_transform(X_sample)

    start = timer()
    lamp = Lamp(nr_neighbors=10).fit(X_sample=X_sample,
                                     y_sample=y_sample)
    y = lamp.transform(X=X)
    end = timer()

    print('Lamp took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


def main_force():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)

    # define sample size
    sample_size = int(math.sqrt(len(X)))
    sample_size = max(sample_size, 2 * len(X[0]))

    # get a random sample
    random.seed(7)
    indexes = random.sample(range(len(X)), sample_size)
    X_sample = np.zeros((sample_size, len(X[0])))
    for i in range(sample_size):
        X_sample[i] = X[indexes[i]]

    start = timer()
    lamp = Lamp().fit(X_sample=X_sample)
    y = lamp.transform(X=X)
    end = timer()

    print('Lamp took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    main_tsne()
    exit(0)
