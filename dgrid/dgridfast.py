import math
import numpy as np

import time
import numba

from sklearn.neighbors import KDTree


@numba.jit(nopython=True, parallel=False)
def _density_calculation(count_map, mask, mask_size, x_min, x_max, y_min, y_max, nr_columns, nr_rows):
    dummy_points_candidates = []

    for row in range(nr_rows):
        y_ = row * (y_max - y_min) / (nr_rows - 1) + y_min

        for column in range(nr_columns):
            if count_map[row][column] == 0:
                x_ = column * (x_max - x_min) / (nr_columns - 1) + x_min
                density = 0

                for i in range(mask_size):
                    for j in range(mask_size):
                        r = row - (int(mask_size / 2)) + j
                        c = column - (int(mask_size / 2)) + i

                        if (0 <= r < nr_rows) and (0 <= c < nr_columns):
                            density += mask[i][j] * count_map[r][c]

                dummy_points_candidates.append([x_, y_, density, -1])

    return dummy_points_candidates


class DGridFast:

    def __init__(self,
                 glyph_width=1,
                 glyph_height=1,
                 delta=None
                 ):
        self.glyph_width_ = glyph_width
        self.glyph_height_ = glyph_height
        self.delta_ = delta

        if self.delta_ is None:
            self.delta_ = 1

        self.grid_ = []

    def _fit(self, y):
        # calculating the bounding box
        max_coordinates = np.amax(y, axis=0)
        min_coordinates = np.amin(y, axis=0)
        bounding_box_width = max_coordinates[0] - min_coordinates[0]
        bounding_box_height = max_coordinates[1] - min_coordinates[1]

        # defining the number of rows and columns
        nr_columns = math.ceil((self.delta_ * bounding_box_width) / self.glyph_width_)
        nr_rows = math.ceil((self.delta_ * bounding_box_height) / self.glyph_height_)

        # if the number of rows and columns are not enough to fit all data instances, increase delta
        if nr_rows * nr_columns < len(y):
            nr_columns = (bounding_box_width / self.glyph_width_)
            nr_rows = (bounding_box_height / self.glyph_height_)
            self.delta_ = math.sqrt(len(y) / (nr_rows * nr_columns))
            nr_columns = math.ceil(self.delta_ * nr_columns)
            nr_rows = math.ceil(self.delta_ * nr_rows)

            print("There is not enough space to remove overlaps! Setting delta to {0}, the smallest possible number "
                  "to fully remove overlaps. Increase it if more empty space is required.".format(self.delta_))

        # add the original points
        def to_grid_cell(id_, x_, y_):
            return {'id': id_,
                    'x': x_,
                    'y': y_,
                    'i': 0,
                    'j': 0,
                    'dummy': False}

        for i in range(len(y)):
            self.grid_.append(to_grid_cell(i, y[i][0], y[i][1]))

        # add the dummy points
        start_time = time.time()
        self._add_dummy_points(min_coordinates[0], max_coordinates[0],
                               min_coordinates[1], max_coordinates[1],
                               nr_columns, nr_rows)
        print("--- Add dummy points executed in %s seconds ---" % (time.time() - start_time))

        # remove overlaps
        start_time = time.time()

        # creating the initial indices
        index_x, index_y = self._init_indices()

        self.grid_ = self._grid_rec(index_x, index_y, nr_rows, nr_columns, 0, 0)
        self.grid_.sort(key=lambda v: v.get('id'))
        print("--- Grid assignment executed in %s seconds ---" % (time.time() - start_time))

        # returning the overlap free scatterplot
        transformed = []
        for i in range(len(self.grid_)):
            if self.grid_[i]['dummy'] is False:
                transformed.append(np.array([min_coordinates[0] + self.grid_[i]['j'] * self.glyph_width_,
                                             min_coordinates[1] + self.grid_[i]['i'] * self.glyph_height_]))

        return np.array(transformed)

    def fit_transform(self, y):
        return self._fit(y)

    def fit(self, y):
        return self._fit(y)

    def _init_indices(self):
        # creating the index ordered by X
        self.grid_.sort(key=lambda cel: (cel['x'], cel['y']))
        index_x = []
        for cell_grid in self.grid_:
            index_x.append([cell_grid['id'], cell_grid['x'], cell_grid['y']])

        # creating the index ordered by Y
        self.grid_.sort(key=lambda cel: (cel['y'], cel['x']))
        index_y = []
        for cell_grid in self.grid_:
            index_y.append([cell_grid['id'], cell_grid['x'], cell_grid['y']])

        # returning to the original ordering
        self.grid_.sort(key=lambda cel: cel['id'])

        return index_x, index_y

    @staticmethod
    def _split_grid(index_x, index_y, cut_point, direction):
        # VERY IMPORTANT
        # second list can be empty, so this needs to be handled in some way
        # VERY IMPORTANT
        if cut_point >= len(index_x):
            return index_x, [], index_y, []
        # VERY IMPORTANT
        # second list can be empty, so this needs to be handled in some way
        # VERY IMPORTANT

        index_x0 = []
        index_x1 = []

        index_y0 = []
        index_y1 = []

        if direction is 'x':
            # splitting the index_x keeping order
            index_x0 = index_x[:cut_point]
            index_x1 = index_x[-(len(index_x) - cut_point):]

            # splitting index_y keeping order
            # getting the coordinates of the element on the cut point
            x_cutpoint = index_x[cut_point - 1][1]
            y_cutpoint = index_x[cut_point - 1][2]

            for index in index_y:
                x_index_y = index[1]
                y_index_y = index[2]

                if x_index_y < x_cutpoint:
                    index_y0.append(index)
                elif x_index_y > x_cutpoint:
                    index_y1.append(index)
                elif x_index_y == x_cutpoint:
                    if y_index_y < y_cutpoint:
                        index_y0.append(index)
                    elif y_index_y > y_cutpoint:
                        index_y1.append(index)
                    else:
                        if len(index_y0) < cut_point:
                            index_y0.append(index)
                        else:
                            index_y1.append(index)
        else:
            # splitting the index_y keeping order
            index_y0 = index_y[:cut_point]
            index_y1 = index_y[-(len(index_y) - cut_point):]

            # splitting index_y keeping order
            # getting the coordinates of the element on the cut point
            x_cutpoint = index_y[cut_point - 1][1]
            y_cutpoint = index_y[cut_point - 1][2]

            for index in index_x:
                x_index_x = index[1]
                y_index_x = index[2]

                if y_index_x < y_cutpoint:
                    index_x0.append(index)
                elif y_index_x > y_cutpoint:
                    index_x1.append(index)
                elif y_index_x == y_cutpoint:
                    if x_index_x < x_cutpoint:
                        index_x0.append(index)
                    elif x_index_x > x_cutpoint:
                        index_x1.append(index)
                    else:
                        if len(index_x0) < cut_point:
                            index_x0.append(index)
                        else:
                            index_x1.append(index)

        return index_x0, index_x1, index_y0, index_y1

    def _grid_rec(self, index_x, index_y, r, s, i, j):
        nr_elements = len(index_x)

        if nr_elements > 0:
            if nr_elements == 1:
                self.grid_[index_x[0][0]]['i'] = i
                self.grid_[index_x[0][0]]['j'] = j
            else:
                if r > s:
                    half_rows = int(math.ceil(r / 2.0))

                    index_x0, index_x1, index_y0, index_y1 = self._split_grid(index_x,
                                                                              index_y,
                                                                              min(nr_elements, half_rows * s),
                                                                              'y')
                    self._grid_rec(index_x0, index_y0, half_rows, s, i, j)
                    self._grid_rec(index_x1, index_y1, (r - half_rows), s, (i + half_rows), j)
                else:
                    half_columns = (math.ceil(s / 2.0))

                    index_x0, index_x1, index_y0, index_y1 = self._split_grid(index_x,
                                                                              index_y,
                                                                              min(nr_elements, half_columns * r),
                                                                              'x')
                    self._grid_rec(index_x0, index_y0, r, half_columns, i, j)
                    self._grid_rec(index_x1, index_y1, r, (s - half_columns), i, (j + half_columns))

        return self.grid_

    def _add_dummy_points(self, x_min, x_max, y_min, y_max, nr_columns, nr_rows):
        size = len(self.grid_)

        # counting the number of points per grid cell
        count_map = np.zeros((nr_rows, nr_columns), dtype=np.uint32)

        for i in range(size):
            col = math.ceil(((self.grid_[i]['x'] - x_min) / (x_max - x_min)) * (nr_columns - 1))
            row = math.ceil(((self.grid_[i]['y'] - y_min) / (y_max - y_min)) * (nr_rows - 1))
            count_map[row][col] = count_map[row][col] + 1

        # calculating the gaussian mask
        mask_size = int(max(3, ((x_max - x_min) * (y_max - y_min)) / (size * self.glyph_width_ * self.glyph_height_)))
        mask_size = mask_size + 1 if mask_size % 2 == 0 else mask_size
        mask = DGridFast._gaussian_mask(mask_size, (mask_size - 1) / 6.0)

        # creating all dummy candidates
        dummy_points_candidates = _density_calculation(count_map, mask, mask_size,
                                                       x_min, x_max, y_min, y_max,
                                                       nr_columns, nr_rows)

        # sorting candidates using density
        dummy_points_candidates.sort(key=lambda x: x[2])

        # defining the number of required dummy points
        nr_dummy_points = min((nr_rows * nr_columns) - size, len(dummy_points_candidates))

        # checking if density is not enough to decide the correct dummy points
        if len(dummy_points_candidates) > nr_dummy_points and math.fabs(
                dummy_points_candidates[nr_dummy_points - 1][2] -
                dummy_points_candidates[nr_dummy_points][2]) < 0.0001:

            # if not, create a kd-tree to find the nearest point in the original layout
            original_points = []
            for i in range(size):
                # adding the original points
                x_ = self.grid_[i]['x']
                y_ = self.grid_[i]['y']
                original_points.append([x_, y_])

            tree = KDTree(original_points, leaf_size=2)

            # add the distance information for the "undecided" dummy points
            for i in range(len(dummy_points_candidates)):
                if math.fabs(dummy_points_candidates[nr_dummy_points - 1][2] -
                             dummy_points_candidates[i][2]) < 0.0001:
                    dummy_points_candidates[i][3] = float(tree.query([[dummy_points_candidates[i][0],
                                                                       dummy_points_candidates[i][1]]], 1)[0])

            # sort the candidates again using density and distance
            dummy_points_candidates.sort(key=lambda x: (x[2], x[3]))

        for i in range(nr_dummy_points):
            self.grid_.append({'id': size + i,
                               'x': dummy_points_candidates[i][0],
                               'y': dummy_points_candidates[i][1],
                               'i': 0,
                               'j': 0,
                               'dummy': True})

        return

    @staticmethod
    def _gaussian_mask(size, sigma):
        mask = np.zeros((size, size), dtype=np.float32)

        for i in range(size):
            y = int(i - int(size / 2))

            for j in range(size):
                x = int(j - int(size / 2))
                mask[i][j] = 1.0 / (2 * math.pi * sigma * sigma) * math.exp(-(x * x + y * y) / (2 * sigma * sigma))

        return mask
