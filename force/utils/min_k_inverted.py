import math
import numpy as np
from numba import njit, prange
from utils.sort import min_k_numba
from utils.distance import get_distance


@njit(
    cache=True,
    parallel=True,
    fastmath=True,
    boundscheck=False,
    nogil=True,
)
def calculate_k_distance(distance_matrix, nr_neighbors):
    total = len(distance_matrix)
    size = int((math.sqrt(1 + 8 * total) - 1) / 2)

    nr_neighbors_2 = nr_neighbors**2
    # adjusting the number of neighbors in case it is larger than the dataset
    nr_neighbors = min(nr_neighbors, size - 1)

    k_distance = np.zeros(size, dtype=np.float32)
    k_distance_idx_f = np.zeros((size, nr_neighbors), dtype=np.int32)
    k_distance_idx_k = np.zeros(size, dtype=np.int32)
    k_distance_idx = -np.ones((size, nr_neighbors_2), dtype=np.int32)

    for ins1 in prange(size):
        distance_array = np.zeros(size, dtype=np.float32)
        for ins2 in range(size):
            if ins1 == ins2:
                continue
            distance_array[ins2] = get_distance(
                distance_matrix, ins1, ins2, total, size
            )
        arr = min_k_numba(distance_array, nr_neighbors + 1)
        k_distance_idx_f[ins1] = arr[0:nr_neighbors]
        dists = distance_array[arr]
        dists.sort()
        k_distance[ins1] = dists[-1]
        # for ins2 in arr[0:nr_neighbors]:
        #     if k_distance_idx_k[ins2] < nr_neighbors_2:
        #         k_distance_idx[ins2, k_distance_idx_k[ins2]] = ins1
        #         k_distance_idx_k[ins2] += 1

    for ins1 in prange(size):
        for ins2 in prange(size):
            if ins1 == ins2:
                continue
            drn = get_distance(distance_matrix, ins1, ins2, total, size)

            if drn <= k_distance[ins2] and k_distance_idx_k[ins1] < nr_neighbors_2:
                k_distance_idx[ins1, k_distance_idx_k[ins1]] = ins2
                k_distance_idx_k[ins1] += 1

    return k_distance_idx, k_distance
