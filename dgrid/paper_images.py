import numpy as np
import pandas as pd

import time
from matplotlib.colors import ListedColormap

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP

import sklearn.datasets as datasets
from sklearn import preprocessing

from dgrid import DGrid
import scatterplot as sct


def main_fig_happiness():
    # read multidimensional data
    data_file = "./data/happines2019.csv"
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

    glyph_size = 0.35

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(glyph_width=glyph_size, glyph_height=glyph_size, delta=1.0).fit_transform(y)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    sct.starglyphs(y_overlap_removed, X, glyph_width=glyph_size,
                   glyph_height=glyph_size, label=labels, names=names,
                   figsize=(25, 11), fontsize=6, alpha=0.75, cmap="cividis")
    sct.title('DGrid Scatterplot')
    sct.savefig("hapiness_dgrid.png", dpi=400)
    sct.show()


def main_fig_cancer():
    # load data
    raw = datasets.load_breast_cancer(as_frame=True)
    X = preprocessing.StandardScaler().fit_transform(raw.data.to_numpy())

    # apply dimensionality reduction
    y = TSNE(n_components=2, random_state=0).fit_transform(X)

    glyph_size = 1.75
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
    y_overlap_removed = DGrid(glyph_width=glyph_size, glyph_height=glyph_size, delta=delta).fit_transform(y)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    cmap = ListedColormap(['#e31a1c', '#aaaaaa'])
    sct.circles(y_overlap_removed, glyph_width=glyph_size, glyph_height=glyph_size, label=raw.target,
                figsize=(11, 11), alpha=1.0, cmap=cmap)
    sct.title('DGrid Scatterplot')
    sct.savefig("breast_cancer-" + str(delta) + ".png", dpi=400)
    sct.show()


def main_fig_fmnist():
    # read multidimensional data
    data_file = "./data/fmnist_test_features.csv"
    df = pd.read_csv(data_file, header=0, sep='[;,]', engine='python')

    labels = df[df.columns[128]]  # get correct classes
    predicted = df[df.columns[130]]  # get predicted classes
    correct = df[df.columns[131]]  # get if the prediction was correct

    # creating a new class when the item was incorrectly classified
    predicted_new = np.full(len(predicted), -1)
    for i in range(len(predicted)):
        if correct[i] == 1:
            predicted_new[i] = predicted[i]

    df = df.drop(['label', 'is_test', 'predicted', 'correct'], axis=1)  # removing the column class
    X = df.values

    # apply dimensionality reduction
    y = UMAP(n_components=2, n_neighbors=7, random_state=5).fit_transform(X)

    # saving projection
    df_proj = pd.DataFrame(y, columns=['x', 'y'])
    df_proj['labels'] = labels
    df_proj['predicted'] = predicted
    df_proj['correct'] = correct
    df_proj.to_csv("fmnist_test_features_proj.csv", sep=',')

    glyph_size = 0.15

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(glyph_width=glyph_size, glyph_height=glyph_size, delta=2.0).fit_transform(y)
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

    sct.circles(y_overlap_removed, glyph_width=glyph_size, glyph_height=glyph_size, label=predicted_new,
                alpha=1.0, cmap=cmap, edgecolor=None, figsize=(10, 10))
    sct.title('DGrid Scatterplot')
    sct.savefig("fmnist.png", dpi=400)
    sct.show()


def main_varying_delta():
    # load data
    input_file = "./data/scatterplot[0037].csv"

    df = pd.read_csv(input_file, header=0, delimiter=",")
    labels = df['label'].values  # getting labels
    width_max = df['width'].max()  # getting the max glyph width
    height_max = df['height'].max()  # getting the max glyph height
    y = df[['ux', 'uy']].values  # getting x and y coordinates

    min_x = df['ux'].min()
    max_x = df['ux'].max()

    min_y = df['uy'].min()
    max_y = df['uy'].max()

    bounding_box_width = max_x - min_x
    bounding_box_height = max_y - min_y

    nr_columns = (bounding_box_width / width_max)
    nr_rows = (bounding_box_height / height_max)
    delta = len(y) / (nr_rows * nr_columns)

    print(bounding_box_width, bounding_box_height)
    print(width_max, height_max)
    print(len(y))

    print("minimum delta:", delta)

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(glyph_width=width_max, glyph_height=height_max, delta=delta).fit_transform(y)
    print("--- DGrid executed in %s seconds ---" % (time.time() - start_time))

    # plot
    sct.circles(y_overlap_removed, glyph_width=width_max, glyph_height=height_max, label=labels,
                cmap='Dark2', figsize=(20, 10))
    sct.title('DGrid Scatterplot Example')
    sct.savefig("scatterplot[0037]-fit" + str(delta) + ".png", dpi=400)
    sct.show()

    return


def main_moons():
    n_samples = 1000
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=7)

    glyph_size = 0.05
    delta = 4.0

    y_overlap_removed = DGrid(glyph_width=glyph_size, glyph_height=glyph_size, delta=delta).\
        fit_transform(noisy_moons[0])

    # plot
    cmap = ListedColormap(['#e31a1c', '#aaaaaa'])
    sct.circles(y_overlap_removed, glyph_width=glyph_size, glyph_height=glyph_size, label=noisy_moons[1],
                figsize=(11, 11), alpha=1.0, cmap=cmap)
    sct.title('DGrid Scatterplot')
    sct.savefig("noise_moons-" + str(delta) + ".png", dpi=400)
    sct.show()


if __name__ == "__main__":
    main_moons()
    exit(0)
