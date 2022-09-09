import math
import numpy as np

import matplotlib
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw_starglyph(x, y, size, data, axes, facecolor, alpha):
    nr_points = len(data)
    increments = 360.0 / nr_points

    path_data = [(mpath.Path.MOVETO, (x + ((size / 2) * math.cos(math.radians(0))) * data[0],
                                      y + ((size / 2) * math.sin(math.radians(0))) * data[0]
                                      ))]

    for i in range(1, nr_points):
        x_ = x + ((size / 2) * math.cos(math.radians((i * increments)))) * data[i]
        y_ = y + ((size / 2) * math.sin(math.radians((i * increments)))) * data[i]
        path_data.append((mpath.Path.LINETO, (x_, y_)))

    path_data.append((mpath.Path.CLOSEPOLY, (x + ((size / 2) * math.cos(math.radians(0))) * data[0],
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


def starglyphs(projection, dataset, glyph_width, glyph_height, label, names=None,
               cmap='Dark2', alpha=1.0, figsize=(5, 5), fontsize=6):
    max_glyph_size = max(glyph_width, glyph_height)
    max_coordinates = np.amax(projection, axis=0)
    min_coordinates = np.amin(projection, axis=0)

    min_label = label.min()
    max_label = label.max()

    # divide every column by its maximum
    max_X = np.amax(dataset, axis=0)
    min_X = np.amin(dataset, axis=0)
    for i in range(len(dataset)):
        for j in (range(len(max_X))):
            dataset[i][j] = (dataset[i][j] - min_X[j]) / (max_X[j] - min_X[j])

    norm = matplotlib.colors.Normalize(vmin=min_label, vmax=max_label)
    color_map = matplotlib.cm.get_cmap(cmap)

    figure, axes = plt.subplots(figsize=figsize)
    plt.axis([min_coordinates[0] - max_glyph_size,
              max_coordinates[0] + max_glyph_size,
              min_coordinates[1] - max_glyph_size,
              max_coordinates[1] + max_glyph_size])

    for i in range(len(projection)):
        x_ = projection[i][0]
        y_ = projection[i][1]
        label_ = label[i]
        glyph_size_ = max_glyph_size
        draw_starglyph(x_, y_, glyph_size_, dataset[i], axes, alpha=alpha, facecolor=color_map(norm(label_)))
        if names is not None:
            plt.text(x_, (y_ + glyph_size_ / 2), names[i], horizontalalignment='center', fontsize=fontsize)

    axes.set_aspect(1)


def circles(projection, glyph_width, glyph_height, label,
            cmap='Dark2', alpha=1.0, figsize=(5, 5), linewidth=0.5, edgecolor='white'):
    max_glyph_size = max(glyph_width, glyph_height)
    max_coordinates = np.amax(projection, axis=0)
    min_coordinates = np.amin(projection, axis=0)

    min_label = label.min()
    max_label = label.max()

    norm = matplotlib.colors.Normalize(vmin=min_label, vmax=max_label)
    color_map = matplotlib.cm.get_cmap(cmap)

    figure, axes = plt.subplots(figsize=figsize)
    plt.axis([min_coordinates[0] - max_glyph_size,
              max_coordinates[0] + max_glyph_size,
              min_coordinates[1] - max_glyph_size,
              max_coordinates[1] + max_glyph_size])

    for i in range(len(projection)):
        x_ = projection[i][0]
        y_ = projection[i][1]
        label_ = label[i]
        glyph_size_ = max_glyph_size

        circle = plt.Circle((x_, y_), (glyph_size_ / 2),
                            linewidth=linewidth,
                            edgecolor=edgecolor,
                            alpha=alpha,
                            facecolor=color_map(norm(label_)))
        axes.add_artist(circle)

    axes.set_aspect(1)


def rectangles(projection, glyph_width, glyph_height, label,
               cmap='Dark2', alpha=1.0, figsize=(5, 5), linewidth=0.5, edgecolor='white'):
    max_glyph_size = max(glyph_width, glyph_height)
    max_coordinates = np.amax(projection, axis=0)
    min_coordinates = np.amin(projection, axis=0)

    min_label = label.min()
    max_label = label.max()

    norm = matplotlib.colors.Normalize(vmin=min_label, vmax=max_label)
    color_map = matplotlib.cm.get_cmap(cmap)

    figure, axes = plt.subplots(figsize=figsize)
    plt.axis([min_coordinates[0] - max_glyph_size,
              max_coordinates[0] + max_glyph_size,
              min_coordinates[1] - max_glyph_size,
              max_coordinates[1] + max_glyph_size])

    for i in range(len(projection)):
        x_ = projection[i][0]
        y_ = projection[i][1]
        label_ = label[i]
        glyph_size_ = max_glyph_size

        rect = plt.Rectangle((x_, y_), glyph_size_, glyph_size_,
                             linewidth=linewidth,
                             edgecolor=edgecolor,
                             alpha=alpha,
                             facecolor=color_map(norm(label_)))
        axes.add_artist(rect)

    axes.set_aspect(1)


def title(text):
    plt.title(text)


def savefig(filename, dpi):
    plt.savefig(filename, dpi=dpi)


def show():
    plt.show()
