# Author: Fernando V. Paulovich -- <fpaulovich@gmail.com>
#
# Copyright (c) 2024 Fernando V. Paulovich
# License: MIT

# If you use this implementation, please cite
# Samuel G. Fadel, Francisco M. Fatore, Felipe S.L.G. Duarte, Fernando V. Paulovich, LoCH: A neighborhood-based
# multidimensional projection technique for high-dimensional sparse spaces, Neurocomputing, Volume 150, Part B, 2015,
# Pages 546-556, https://doi.org/10.1016/j.neucom.2014.07.071.

import sklearn.datasets as datasets
import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
import pandas as pd
import random

from loch import LoCH
from sklearn.manifold import TSNE


def main_no_sample_projection():
    raw = datasets.load_digits(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    start = timer()
    y = LoCH(n_neighbors=10).fit_transform(X=X)
    end = timer()

    print('LSP took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c=raw.target,
                cmap='Set1', edgecolors='face', linewidths=0.5, s=4)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    main_no_sample_projection()
    exit(0)
