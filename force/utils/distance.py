# %%
import numba
import numpy as np
from timeit import default_timer as timer
import math

FLOAT_TYPE = np.float32
FLOAT_BUFFER = np.finfo(FLOAT_TYPE).resolution
# %%


@numba.njit(
    cache=True,
    fastmath=True,
    boundscheck=False,
    nogil=True,
)
def get_distance(distance_matrix, ins1, ins2, total, size):
    r = (ins1 + ins2 - math.fabs(ins1 - ins2)) / 2  # min(i,j)
    s = (ins1 + ins2 + math.fabs(ins1 - ins2)) / 2  # max(i,j)
    return distance_matrix[int(total - ((size - r) * (size - r + 1) / 2) + (s - r))]


@numba.njit(
    cache=True,
    fastmath=True,
    boundscheck=False,
    nogil=True,
)
def euclidean_distance_numba(a, b):
    dist = 0.0
    for c in range(a.shape[0]):
        dist += (b[c] - a[c]) ** 2
    return dist


# %%
if __name__ == "__main__":
    a = np.random.rand(3)
    b = np.random.rand(3)

    start = timer()
    for i in range(10000):
        r1 = np.linalg.norm(a - b)
    end = timer()
    print(end - start)

    start = timer()
    for i in range(10000):
        r2 = euclidean_distance_numba(a, b)
    end = timer()
    print(end - start)
