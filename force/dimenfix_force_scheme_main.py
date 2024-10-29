import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta
from sklearn import preprocessing
from force.dimenfix_force_scheme import DimenFixForceScheme


def main():
    raw = datasets.load_breast_cancer(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.MinMaxScaler().fit_transform(X)

    label = np.array(raw.target).reshape(-1, 1)
    label = preprocessing.MinMaxScaler().fit_transform(label)

    # fixed_feature = label[:, 0]
    fixed_feature = X[:, 5]

    start = timer()
    y = DimenFixForceScheme(max_it=1000, fixed_feature=fixed_feature, alpha=1.0).fit_transform(X)
    end = timer()

    print(np.amin(y, axis=0))

    print('ForceScheme took {0} to execute'.format(timedelta(seconds=end - start)))

    plt.figure()
    plt.scatter(y[:, 1], y[:, 0], c=fixed_feature,
                cmap='viridis', edgecolors='face', linewidths=0.5, s=12)
    plt.grid(linestyle='dotted')
    plt.show()

    return


if __name__ == "__main__":
    main()
    exit(0)
