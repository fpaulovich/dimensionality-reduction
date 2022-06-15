import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn.datasets as datasets
import time

from matplotlib.colors import ListedColormap
from matplotlib import cm

from force_scheme import ForceScheme
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn import preprocessing
from dgrid import DGrid


def plot(y, icon_width, icon_height, label, cmap='Dark2'):
    max_icon_size = max(icon_width, icon_height)
    max_coordinates = np.amax(y, axis=0)
    min_coordinates = np.amin(y, axis=0)

    min_label = label.min()
    max_label = label.max()

    norm = matplotlib.colors.Normalize(vmin=min_label, vmax=max_label)
    color_map = matplotlib.cm.get_cmap(cmap)

    figure, axes = plt.subplots()
    plt.axis([min_coordinates[0] - max_icon_size,
              max_coordinates[0] + max_icon_size,
              min_coordinates[1] - max_icon_size,
              max_coordinates[1] + max_icon_size])

    for i in range(len(y)):
        x_ = y[i][0]
        y_ = y[i][1]
        label_ = label[i]
        icon_size_ = max_icon_size

        circle = plt.Circle((x_, y_), (icon_size_ / 2), linewidth=0.5,
                            edgecolor='white', facecolor=color_map(norm(label_)))
        axes.add_artist(circle)

        # rect = plt.Rectangle((x_, y_), icon_size_, icon_size_, linewidth=0.5,
        #                      edgecolor='white', facecolor=color_map(norm(label_)))
        # axes.add_artist(rect)

    axes.set_aspect(1)


def main1():
    # load data
    raw = datasets.load_iris(as_frame=True)
    X = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)

    # apply dimensionality reduction
    # y = TSNE(n_components=2).fit_transform(X)
    y = ForceScheme().fit_transform(X)
    # y = PCA(n_components=2).fit_transform(X)

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(icon_width=1, icon_height=1, delta=25).fit_transform(y)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    plot(y_overlap_removed, icon_width=1, icon_height=1, label=raw.target, cmap='Dark2')
    plt.title('DGrid Scatterplot')
    # plt.savefig("transformed.png", dpi=400)
    plt.show()

    return


def main2():
    # load data
    input_file = "scatterplot[0001].csv"

    df = pd.read_csv(input_file, header=0, delimiter=",")
    labels = df['label'].values  # getting labels
    width_max = df['width'].max()  # getting the max icon width
    height_max = df['height'].max()  # getting the max icon height
    y = df[['ux', 'uy']].values  # getting x and y coordinates

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(icon_width=width_max, icon_height=height_max, delta=1).fit_transform(y)
    print("--- DGrid executed in %s seconds ---" % (time.time() - start_time))

    # plot
    plot(y_overlap_removed, icon_width=width_max, icon_height=height_max, label=labels, cmap='Dark2')
    plt.title('DGrid Scatterplot')
    # plt.savefig("transformed.png", dpi=400)
    plt.show()

    return


if __name__ == "__main__":
    main2()
    exit(0)
