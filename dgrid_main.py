import math

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

import matplotlib.path as mpath
import matplotlib.patches as mpatches

from umap import UMAP


def draw_starglyph(x, y, size, data, axes, facecolor, alpha):
    nr_points = len(data)
    increments = 360.0 / nr_points

    Path = mpath.Path
    path_data = [(Path.MOVETO, (x + ((size / 2) * math.cos(math.radians(0))) * data[0],
                                y + ((size / 2) * math.sin(math.radians(0))) * data[0]
                                ))]

    for i in range(1, nr_points):
        x_ = x + ((size / 2) * math.cos(math.radians((i * increments)))) * data[i]
        y_ = y + ((size / 2) * math.sin(math.radians((i * increments)))) * data[i]
        path_data.append((Path.LINETO, (x_, y_)))

    path_data.append((Path.CLOSEPOLY, (x + ((size / 2) * math.cos(math.radians(0))) * data[0],
                                       y + ((size / 2) * math.sin(math.radians(0))) * data[0]
                                       )))

    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=facecolor, linewidth=0.5, edgecolor='black', alpha=alpha)

    axes.add_patch(patch)

    ###############

    Path = mpath.Path
    path_data = []

    for i in range(nr_points):
        path_data.append((Path.MOVETO, (x, y)))
        x_ = x + (size / 2) * math.cos(math.radians((i * increments)))
        y_ = y + (size / 2) * math.sin(math.radians((i * increments)))
        path_data.append((Path.LINETO, (x_, y_)))
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, linewidth=0.35, edgecolor='black', linestyle=':', alpha=0.5)

    axes.add_patch(patch)


def plot_starglyphs(y, X, icon_width, icon_height, label, names=None,
                    cmap='Dark2', figsize=(5, 5), fontsize=6, alpha=1.0):
    max_icon_size = max(icon_width, icon_height)
    max_coordinates = np.amax(y, axis=0)
    min_coordinates = np.amin(y, axis=0)

    min_label = label.min()
    max_label = label.max()

    # divide every column by its maximum
    max_X = np.amax(X, axis=0)
    min_X = np.amin(X, axis=0)
    for i in range(len(X)):
        for j in (range(len(max_X))):
            X[i][j] = (X[i][j] - min_X[j]) / (max_X[j] - min_X[j])

    norm = matplotlib.colors.Normalize(vmin=min_label, vmax=max_label)
    color_map = matplotlib.cm.get_cmap(cmap)

    figure, axes = plt.subplots(figsize=figsize)
    plt.axis([min_coordinates[0] - max_icon_size,
              max_coordinates[0] + max_icon_size,
              min_coordinates[1] - max_icon_size,
              max_coordinates[1] + max_icon_size])

    for i in range(len(y)):
        x_ = y[i][0]
        y_ = y[i][1]
        label_ = label[i]
        icon_size_ = max_icon_size
        draw_starglyph(x_, y_, icon_size_, X[i], axes, alpha=alpha, facecolor=color_map(norm(label_)))
        if names is not None:
            plt.text(x_, (y_ + icon_size_ / 2), names[i], horizontalalignment='center', fontsize=fontsize)

    axes.set_aspect(1)


def plot_circles(y, icon_width, icon_height, label, cmap='Dark2', alpha=1.0, figsize=(5, 5), edgecolor='white'):
    max_icon_size = max(icon_width, icon_height)
    max_coordinates = np.amax(y, axis=0)
    min_coordinates = np.amin(y, axis=0)

    min_label = label.min()
    max_label = label.max()

    norm = matplotlib.colors.Normalize(vmin=min_label, vmax=max_label)
    color_map = matplotlib.cm.get_cmap(cmap)

    figure, axes = plt.subplots(figsize=figsize)
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
                            edgecolor=edgecolor, alpha=alpha, facecolor=color_map(norm(label_)))
        axes.add_artist(circle)

        # rect = plt.Rectangle((x_, y_), icon_size_, icon_size_, linewidth=0.5,
        #                      edgecolor='white', facecolor=color_map(norm(label_)))
        # axes.add_artist(rect)

    axes.set_aspect(1)


def main1():
    # load data
    raw = datasets.load_iris(as_frame=True)
    X_orig = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(raw.data.to_numpy())

    # apply dimensionality reduction
    # y = TSNE(n_components=2, random_state=0).fit_transform(X)
    # y = ForceScheme().fit_transform(X)
    y = PCA(n_components=2).fit_transform(X)

    icon_size = 0.25
    delta = 1.0

    # remove overlaps
    start_time = time.time()
    # y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=delta).fit_transform(y) # digits (icon_size=1.75)
    # y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=delta).fit_transform(y)  # breast cancer (icon_size=1.75)
    y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=delta).fit_transform(
        y)  # iris(icon_size=0.25)
    # y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=delta).fit_transform(y)  # wine(icon_size=1.0)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    plot_starglyphs(y_overlap_removed, X_orig, icon_width=icon_size, icon_height=icon_size, label=raw.target,
                    cmap="Set1")  # Dark2_r
    plt.title('DGrid Scatterplot')
    # plt.savefig("/Users/fpaulovich/Desktop/breast_cancer-" + str(delta) + ".png", dpi=400)
    plt.show()

    return


def main2():
    # load data
    input_file = "/Users/fpaulovich/Dropbox/Papers/2021/DGrid/tests/final_tests/original/scatterplot[0047].csv"

    df = pd.read_csv(input_file, header=0, delimiter=",")
    labels = df['label'].values  # getting labels
    width_max = df['width'].max()  # getting the max icon width
    height_max = df['height'].max()  # getting the max icon height
    y = df[['ux', 'uy']].values  # getting x and y coordinates

    # remove overlaps
    delta = 2.0

    start_time = time.time()
    y_overlap_removed = DGrid(icon_width=width_max, icon_height=height_max, delta=delta).fit_transform(y)
    print("--- DGrid executed in %s seconds ---" % (time.time() - start_time))

    # plot
    plot_circles(y_overlap_removed, icon_width=width_max, icon_height=height_max, label=labels, cmap='Dark2')
    plt.title('DGrid Scatterplot')
    plt.savefig("/Users/fpaulovich/Desktop/scatterplot[0047]-" + str(delta) + ".png", dpi=400)
    plt.show()

    return


def main3():
    input_file = "/Users/fpaulovich/Dropbox/datasets/csv/cbr-ilp-ir.csv"
    df = pd.read_csv(input_file, header=0, sep='[;,]', engine='python')

    labels = df[df.columns[len(df.columns) - 1]]  # get the last column as labels
    df = df.drop(labels='label', axis=1)  # removing the column class
    df = df.drop(labels='id', axis=1)  # removing the id class

    X = preprocessing.StandardScaler().fit_transform(df.values)
    # X = df.values
    # y = ForceScheme().fit_transform(X)
    y = TSNE(n_components=2, random_state=0).fit_transform(X)

    icon_size = 2.25

    # remove overlaps
    start_time = time.time()
    # y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=2).fit_transform(y) # cbr-ilp-ir (icon=2.25)
    # y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=1).fit_transform(y) # cbrilpirson (icon=2.0)
    y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=1).fit_transform(
        y)  # cbrilpirson (icon=2.0)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    # plot(y_overlap_removed, icon_width=icon_size, icon_height=icon_size, label=labels, cmap='Dark2')
    plot_starglyphs(y_overlap_removed, df.values, icon_width=icon_size, icon_height=icon_size, label=labels,
                    cmap="Dark2")
    plt.title('DGrid Scatterplot')
    plt.savefig("/Users/fpaulovich/Desktop/cbr-ilp-ir-2.0.png", dpi=400)
    plt.show()


def main4():
    # read multidimensional data
    data_file = "/Users/fpaulovich/Dropbox/datasets/csv/happines2019.csv"
    df = pd.read_csv(data_file, header=0, sep='[;,]', engine='python')

    names = df[df.columns[1]]  # get country names
    labels = df[df.columns[2]]  # get scores

    df = df.drop(labels='Overall rank', axis=1)  # removing the column class
    df = df.drop(labels='Country or region', axis=1)  # removing the id class
    df = df.drop(labels='Score', axis=1)  # removing the id class
    X = df.values  # preprocessing.MinMaxScaler().fit_transform(df.values)

    # read projection
    projection_file = "/Users/fpaulovich/Dropbox/Papers/2021/DGrid/dgrid/teaser/hapiness/happiness_umap_4_1_noScore_notNorm_overlap.csv"
    df = pd.read_csv(projection_file, header=0, delimiter=",")
    y = df[['ux', 'uy']].values  # getting x and y coordinates

    icon_size = 250

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=1.75).fit_transform(y)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    plot_starglyphs(y_overlap_removed, X, icon_width=icon_size,
                    icon_height=icon_size, label=labels, names=names,
                    figsize=(25, 11), fontsize=6, cmap="cividis")

    plt.title('DGrid Scatterplot')
    plt.savefig("/Users/fpaulovich/Desktop/hapiness.png", dpi=400)
    plt.show()


def main_fig_happiness():
    # read multidimensional data
    data_file = "/Users/fpaulovich/Dropbox/datasets/csv/happines2019.csv"
    df = pd.read_csv(data_file, header=0, sep='[;,]', engine='python')

    names = df[df.columns[1]]  # get country names
    labels = df[df.columns[2]]  # get scores

    # trunk names size
    for i in range(len(names)):
        names[i] = names[i][:9]

    df = df.drop(['Score', 'Overall rank', 'Country or region'], axis=1)  # removing the column class
    X = df.values

    # apply dimensionality reduction
    y = UMAP(n_components=2, n_neighbors=7, random_state=5).fit_transform(X)

    # rotate
    pca = PCA(n_components=2)
    pca.fit(y)
    y = np.dot(y, pca.components_)

    icon_size = 0.35

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=1.0).fit_transform(y)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    plot_starglyphs(y_overlap_removed, X, icon_width=icon_size,
                    icon_height=icon_size, label=labels, names=names,
                    figsize=(25, 11), fontsize=6, alpha=0.75, cmap="cividis")
    plt.title('DGrid Scatterplot')
    plt.savefig("/Users/fpaulovich/Desktop/hapiness_dgrid.png", dpi=400)
    plt.show()


def main_fig_cancer():
    # load data
    raw = datasets.load_breast_cancer(as_frame=True)
    X = preprocessing.StandardScaler().fit_transform(raw.data.to_numpy())

    # apply dimensionality reduction
    y = TSNE(n_components=2, random_state=0).fit_transform(X)

    icon_size = 1.75
    delta = 1.0

    # sort points according to target
    def to_point(x_, y_, label_):
        return {'x': x_,
                'y': y_,
                'label': label_}

    points = []
    for i in range(len(y)):
        points.append(to_point(y[i][0], y[i][1], raw.target[i]))
    points.sort(key=lambda v: v.get('label'))

    for i in range(len(y)):
        y[i][0] = points[i]['x']
        y[i][1] = points[i]['y']
        raw.target[i] = points[i]['label']

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=delta).fit_transform(y)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    cmap = ListedColormap(['#6a3d9a', '#ff7f00'])
    plot_circles(y_overlap_removed, icon_width=icon_size, icon_height=icon_size, label=raw.target,
                 alpha=0.95, cmap=cmap)
    plt.title('DGrid Scatterplot')
    plt.savefig("/Users/fpaulovich/Desktop/breast_cancer-" + str(delta) + ".png", dpi=400)
    plt.show()

    return


def main_fig_fmnist():
    # read multidimensional data
    data_file = "/Users/fpaulovich/Dropbox/datasets/csv/fmnist_test_features.csv"
    df = pd.read_csv(data_file, header=0, sep='[;,]', engine='python')

    labels = df[df.columns[128]]  # get correct classes
    predicted = df[df.columns[130]]  # get predicted classes
    correct = df[df.columns[131]]  # get if the prediction was correct

    # creating a new class when the item was incorrectly classified
    for i in range(len(predicted)):
        if correct[i] == 0:
            predicted[i] = -1

    df = df.drop(['label', 'is_test', 'predicted', 'correct'], axis=1)  # removing the column class
    X = df.values

    # apply dimensionality reduction
    y = UMAP(n_components=2, n_neighbors=7, random_state=5).fit_transform(X)

    icon_size = 0.15

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(icon_width=icon_size, icon_height=icon_size, delta=2.0).fit_transform(y)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    cmap = ListedColormap([
        '#e31a1c',
        '#8dd3c7',
        '#bebada',
        '#80b1d3',
        '#fdb462',
        '#b3de69',
        '#fccde5',
        '#d9d9d9',
        '#bc80bd',
        '#ccebc5',
        '#ffed6f'
    ])

    plot_circles(y_overlap_removed, icon_width=icon_size, icon_height=icon_size, label=predicted,
                 alpha=1.0, cmap=cmap, edgecolor=None, figsize=(10, 10))
    plt.title('DGrid Scatterplot')
    plt.savefig("/Users/fpaulovich/Desktop/fmnist.png", dpi=400)
    plt.show()


if __name__ == "__main__":
    main_fig_fmnist()
    exit(0)
