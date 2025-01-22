# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright 2024 Fernando V. Paulovich
# License: BSD-3-Clause

# If you use this implementation, please cite
# F. V. Paulovich and R. Minghim, "Text Map Explorer: a Tool to Create and Explore Document Maps," Tenth International
# Conference on Information Visualisation (IV'06), London, UK, 2006, pp. 245-251, doi: 10.1109/IV.2006.104.

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from pbc import PBC


def main():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    start = timer()
    y = PBC(cluster_factor=0.2).fit_transform(X)
    end = timer()

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
