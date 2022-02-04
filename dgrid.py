import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import matplotlib
from matplotlib import cm
import numpy as np
from matplotlib.colors import ListedColormap


class DistanceGrid:

    @staticmethod
    def execute(data, x='ux', y='uy', width='width', height='height', label='label', delta=1):
        icon_size_width, icon_size_height = DistanceGrid.max_icon_dimension(data, width, height)
        bounding_box_width, bounding_box_height = DistanceGrid.bounding_box(data, x, y, width, height)

        nr_columns = int((delta * bounding_box_width) / icon_size_width)
        nr_rows = int((delta * bounding_box_height) / icon_size_height)

        if (nr_rows * nr_columns) < len(data):
            raise Exception("There is no space to remove overlaps! Rows: {0}, columns: {1}, data size: {2}. Try "
                            "increasing delta.".
                            format(nr_rows, nr_columns, len(data)))

        # add the original points
        def to_grid_cell(id_, x_, y_):
            return {'id': id_,
                    'x': x_,
                    'y': y_,
                    'i': 0,
                    'j': 0,
                    'dummy': False}

        grid = []

        for i in range(len(data)):
            grid.append(to_grid_cell(i, data.iloc[i][x], data.iloc[i][y]))

        # add the dummy points
        DistanceGrid.add_dummy_points(grid, icon_size_width, icon_size_height, nr_rows, nr_columns)

        # execute
        grid = DistanceGrid.grid_rec(grid, nr_rows, nr_columns, 0, 0)
        grid.sort(key=lambda v: v.get('id'))

        transformed = []
        for i in range(len(grid)):
            if grid[i]['dummy'] is False:
                transformed.append([grid[i]['j'] * icon_size_width,
                                    grid[i]['i'] * icon_size_height,
                                    icon_size_width,
                                    icon_size_height,
                                    data.iloc[i][label]])

        return pd.DataFrame(transformed, columns=['ux', 'uy', 'width', 'height', 'label'])

    @staticmethod
    def add_dummy_points(grid, icon_size_width, icon_size_height, nr_rows, nr_columns):
        size = len(grid)
        min_x = min_y = math.inf
        max_x = max_y = -math.inf

        scatterplot = []

        for i in range(size):
            x_ = grid[i]['x']
            y_ = grid[i]['y']

            scatterplot.append([x_, y_])

            if min_x > x_:
                min_x = x_

            if min_y > y_:
                min_y = y_

            if max_x < x_:
                max_x = x_

            if max_y < y_:
                max_y = y_

        count = [[0] * nr_columns for _ in range(nr_rows)]

        for i in range(size):
            col = int((grid[i]['x'] - min_x) / icon_size_width)
            row = int((grid[i]['y'] - min_y) / icon_size_height)
            count[row][col] = count[row][col] + 1

        icons_area = size * icon_size_width * icon_size_height
        scatterplot_area = (max_x - min_x) * (max_y - min_y)
        mask_size = int(max(3, scatterplot_area / icons_area))
        mask_size = mask_size + 1 if mask_size % 2 == 0 else mask_size

        sigma = (mask_size - 1) / 6.0
        mask = DistanceGrid.gaussian_mask(mask_size, sigma)

        # create a kd-tree to calculate the closest points
        tree = KDTree(scatterplot, leaf_size=2)

        dummy_points_candidates = []

        for row in range(nr_rows):
            y_ = row * (max_y - min_y) / (nr_rows - 1) + min_y

            for column in range(nr_columns):
                if count[row][column] == 0:
                    density = 0

                    for i in range(mask_size):
                        for j in range(mask_size):
                            r = row - (int(mask_size / 2)) + j
                            c = column - (int(mask_size / 2)) + i

                            if (0 <= r < nr_rows) and (0 <= c < nr_columns):
                                density += mask[i][j] * count[r][c]

                    x_ = column * (max_x - min_x) / (nr_columns - 1) + min_x

                    distance, index = tree.query([[x_, y_]], 1)

                    dummy_points_candidates.append([x_, y_, density, distance[0][0]])

        dummy_points_candidates.sort(key=lambda x: (x[2], x[3]))

        nr_dummy_points = min((nr_rows * nr_columns) - size, len(dummy_points_candidates))

        for i in range(nr_dummy_points):
            grid.append({'id': size + i,
                         'x': dummy_points_candidates[i][0],
                         'y': dummy_points_candidates[i][1],
                         'i': 0,
                         'j': 0,
                         'dummy': True})

        return

    @staticmethod
    def gaussian_mask(size, sigma):
        mask = [[0.0] * size for _ in range(size)]

        for i in range(size):
            y = int(i - int(size / 2))

            for j in range(size):
                x = int(j - int(size / 2))
                mask[i][j] = 1.0 / (2 * math.pi * sigma * sigma) * math.exp(-(x * x + y * y) / (2 * sigma * sigma))

        return mask

    @staticmethod
    def max_icon_dimension(data, width, height):
        max_icon_size_width = data.max()[width]
        max_icon_size_height = data.max()[height]
        return max_icon_size_width, max_icon_size_height

    @staticmethod
    def bounding_box(data, x, y, width, height):
        min_x = min_y = math.inf
        max_x = max_y = -math.inf

        for i in range(len(data)):
            x_ = data.iloc[i][x]
            y_ = data.iloc[i][y]
            width_ = data.iloc[i][width]
            height_ = data.iloc[i][height]

            if min_x > x_:
                min_x = x_

            if min_y > y_:
                min_y = y_

            if max_x < x_ + width_:
                max_x = x_ + width_

            if max_y < y_ + height_:
                max_y = y_ + height_

        return (max_x - min_x), (max_y - min_y)

    @staticmethod
    def split_grid(grid, cut_point, direction):
        if direction == 'x':
            grid.sort(key=lambda cel: (cel['x'], cel['y']))
        else:
            grid.sort(key=lambda cel: (cel['y'], cel['x']))

        grid0 = grid[:cut_point]
        grid1 = []
        if cut_point < len(grid):
            grid1 = grid[-(len(grid) - cut_point):]

        return grid0, grid1

    @staticmethod
    def grid_rec(grid, r, s, i, j):
        size = len(grid)

        if size > 0:
            if size == 1:
                grid[0]['i'] = i
                grid[0]['j'] = j
            else:
                if r > s:
                    half_rows = int(math.ceil(r / 2.0))
                    grid0, grid1 = DistanceGrid.split_grid(grid, min(size, half_rows * s), 'y')
                    DistanceGrid.grid_rec(grid0, half_rows, s, i, j)
                    DistanceGrid.grid_rec(grid1, (r - half_rows), s, (i + half_rows), j)
                else:
                    half_columns = int(math.ceil(s / 2.0))
                    grid0, grid1 = DistanceGrid.split_grid(grid, min(size, half_columns * r), 'x')
                    DistanceGrid.grid_rec(grid0, r, half_columns, i, j)
                    DistanceGrid.grid_rec(grid1, r, (s - half_columns), i, (j + half_columns))

        return grid

    @staticmethod
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
