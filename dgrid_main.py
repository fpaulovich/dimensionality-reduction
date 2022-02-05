import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.colors import ListedColormap
from matplotlib import cm


def plot(data, x, y, width, height, label, cmap='Set2'):
    max_icon_size_width = data.max()[width]
    max_icon_size_height = data.max()[height]
    max_icon_size = max(max_icon_size_width, max_icon_size_height)

    max_width = data.max()[x]
    max_height = data.max()[y]

    min_label = label.min()
    max_label = label.max()

    norm = matplotlib.colors.Normalize(vmin=min_label, vmax=max_label)
    color_map = matplotlib.cm.get_cmap(cmap)

    figure, axes = plt.subplots()
    plt.axis([-max_icon_size, max_width + max_icon_size, -max_icon_size, max_height + max_icon_size])

    for i in range(len(data)):
        x_ = data.iloc[i][x]
        y_ = data.iloc[i][y]
        label_ = label[i]
        icon_size_ = min(data.iloc[i][width], data.iloc[i][height])

        circle = plt.Circle((x_, y_), (icon_size_ / 2) * 0.75, color=color_map(norm(label_)))
        axes.add_artist(circle)

    axes.set_aspect(1)


def main1():
    number = '[0035]'

    input_file = "/Users/fpaulovich/Dropbox/Papers/2021/DGrid/tests/" \
                 "final_tests/original/scatterplot" + number + ".csv"

    original = pd.read_csv(input_file)
    # DistanceGrid.plot(original, 'ux', 'uy', 'width', 'height', original.loc[:, 'label'], cmap='Set2')
    # plt.title('Original Scatterplot')
    # plt.savefig("/Users/paulovich/Desktop/original.png", dpi=400)
    # # plt.show()

    transformed = DistanceGrid.execute(original, delta=1)
    DistanceGrid.plot(transformed, 'ux', 'uy', 'width', 'height', original.loc[:, 'label'], cmap='Set2')
    plt.title('DGrid Scatterplot')
    plt.savefig("/Users/fpaulovich/Desktop/transformed.png", dpi=400)
    # plt.show()

    return


def main2():
    input_file = "/Users/fpaulovich/Documents/protein_folding/matriz_distancia_projection.txt"
    # input_file = "/Users/fpaulovich/Documents/protein_folding/dmat_8000_projection.txt"
    projection = pd.read_csv(input_file, sep=' ', header=None)
    projection = projection.rename(columns={0: "ux", 1: "uy"})
    projection["width"] = 1
    projection["height"] = 1

    labels = np.zeros(len(projection))

    for i in range(8000):
        labels[i] = 0

    for i in range(8000, 16000):
        labels[i] = 1

    for i in range(16000, 25240):
        labels[i] = 2

    projection["label"] = labels

    projection.to_csv('/Users/fpaulovich/Desktop/matriz_distancia_projection_dgrid.csv')

    transformed = DistanceGrid.execute(projection, delta=500)
    DistanceGrid.plot(transformed, 'ux', 'uy', 'width', 'height', transformed.loc[:, 'label'],
                      cmap=ListedColormap(['blue', 'red', 'green']))
    plt.title('DGrid Scatterplot')
    plt.savefig('/Users/fpaulovich/Desktop/transformed.png', dpi=400)
    # plt.show()

    return


if __name__ == "__main__":
    main1()
    exit(0)
