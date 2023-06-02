import pandas as pd

import time

from force.force_scheme import ForceScheme
from sklearn.manifold import TSNE

import sklearn.datasets as datasets
from sklearn import preprocessing
from dgrid import DGrid

import scatterplot as sct


def main1():
    # load data
    raw = datasets.load_iris(as_frame=True)
    X_orig = raw.data.to_numpy()
    X = preprocessing.StandardScaler().fit_transform(raw.data.to_numpy())

    # apply dimensionality reduction
    y = ForceScheme().fit_transform(X)

    glyph_size = 1.0
    delta = 15.0

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(glyph_width=glyph_size, glyph_height=glyph_size,
                              delta=delta, return_type='coord').fit_transform(y)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    sct.starglyphs(y_overlap_removed, X_orig, glyph_width=glyph_size, glyph_height=glyph_size, label=raw.target,
                   cmap="Set2", figsize=(10, 10), alpha=0.75)
    sct.title('DGrid IRIS Scatterplot')
    # sct.savefig("iris-" + str(delta) + ".png", dpi=400)
    sct.show()

    return


def main2():
    # load data
    input_file = "../data/scatterplot.csv"

    df = pd.read_csv(input_file, header=0, delimiter=",")
    labels = df['label'].values  # getting labels
    width_max = df['width'].max()  # getting the max glyph width
    height_max = df['height'].max()  # getting the max glyph height
    y = df[['ux', 'uy']].values  # getting x and y coordinates

    # increase visible area (decrease glyphs sizes)
    delta = 2.0

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(glyph_width=width_max, glyph_height=height_max, delta=delta).fit_transform(y)
    print("--- DGrid executed in %s seconds ---" % (time.time() - start_time))

    # plot
    sct.circles(y_overlap_removed, glyph_width=width_max, glyph_height=height_max, label=labels,
                cmap='Dark2', figsize=(20, 10))
    sct.title('DGrid Scatterplot Example')
    # sct.savefig("scatterplot-" + str(delta) + ".png", dpi=400)
    sct.show()

    return


def main3():
    input_file = "../data/cbr-ilp-ir.csv"
    df = pd.read_csv(input_file, header=0, sep='[;,]', engine='python')

    labels = df[df.columns[len(df.columns) - 1]]  # get the last column as labels
    df = df.drop(labels='label', axis=1)  # removing the column class
    df = df.drop(labels='id', axis=1)  # removing the id class

    X = preprocessing.StandardScaler().fit_transform(df.values)
    y = TSNE(n_components=2, random_state=0).fit_transform(X)

    # setting glyph size and area increase
    glyph_size = 1.5
    delta = 1.0

    # remove overlaps
    start_time = time.time()
    y_overlap_removed = DGrid(glyph_width=glyph_size, glyph_height=glyph_size, delta=delta).fit_transform(y)
    print("--- DGrid execution %s seconds ---" % (time.time() - start_time))

    # plot
    sct.circles(y_overlap_removed, glyph_width=glyph_size, glyph_height=glyph_size, label=labels,
                cmap='Dark2', figsize=(20, 10))
    sct.title('DGrid CBR-ILP-IR Scatterplot')
    # sct.savefig("cbr-ilp-ir.png", dpi=400)
    sct.show()


if __name__ == "__main__":
    main1()
    exit(0)
