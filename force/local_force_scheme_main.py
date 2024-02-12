import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from force.local_force_scheme import LocalForceScheme

import seaborn as sns


def main():
    raw = datasets.load_digits(as_frame=True)

    X = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)

    start = timer()
    y = LocalForceScheme(max_it=500, prob_threshold=0.01, nr_neighbors=15).fit_transform(X)
    end = timer()

    print(np.amin(y, axis=0))

    print('ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

    fig, ax = plt.subplots()
    scatter = ax.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='tab10', edgecolors='face', linewidths=0.5, s=4)
    ax.grid(linestyle='dotted')

    legend1 = ax.legend(*scatter.legend_elements(), loc="lower right", title="class")
    ax.add_artist(legend1)

    plt.show()

    return


if __name__ == "__main__":
    main()
    exit(0)
