import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn.datasets as datasets

from matplotlib.colors import ListedColormap
from matplotlib import cm

from force_scheme import ForceScheme

from sklearn import preprocessing
from dgrid import DGrid


def plot(y, icon_width, icon_height, label, cmap='Dark2'):
    max_icon_size = max(icon_width, icon_height)

    max_width = np.amax(y, axis=0)[0]
    max_height = np.amax(y, axis=0)[1]

    min_label = label.min()
    max_label = label.max()

    norm = matplotlib.colors.Normalize(vmin=min_label, vmax=max_label)
    color_map = matplotlib.cm.get_cmap(cmap)

    figure, axes = plt.subplots()
    plt.axis([-max_icon_size, max_width + max_icon_size, -max_icon_size, max_height + max_icon_size])

    for i in range(len(y)):
        x_ = y[i][0]
        y_ = y[i][1]
        label_ = label[i]
        icon_size_ = max_icon_size

        circle = plt.Circle((x_, y_), (icon_size_ / 2) * 0.75, color=color_map(norm(label_)))
        axes.add_artist(circle)

    axes.set_aspect(1)


def main():
    # load data
    raw = datasets.load_wine(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)

    # apply dimensionality reduction
    y = ForceScheme().fit_transform(X)
    y_overlap_removed = DGrid(icon_width=1, icon_height=1, delta=2).fit_transform(y)

    # plot
    plot(y_overlap_removed, icon_width=1, icon_height=1, label=raw.target, cmap='Dark2')
    plt.title('DGrid Scatterplot')
    # plt.savefig("transformed.png", dpi=400)
    plt.show()

    return


if __name__ == "__main__":
    main()
    exit(0)
