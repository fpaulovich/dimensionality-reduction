import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from force.force_scheme import ForceScheme


def main():
    raw = datasets.load_wine(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)

    start = timer()
    y = ForceScheme(max_it=1000, n_components=3).fit_transform(X)
    end = timer()

    print(np.amin(y, axis=0))

    print('ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(y[:, 0], y[:, 1], y[:, 2], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    main()
    exit(0)
