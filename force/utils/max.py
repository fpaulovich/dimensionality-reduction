# %%
import numba
import numpy as np
from timeit import default_timer as timer

FLOAT_TYPE = np.float32
FLOAT_BUFFER = np.finfo(FLOAT_TYPE).resolution
# %%


@numba.jit(
    cache=True,
    nopython=True,
    parallel=True,
    fastmath=False,
    boundscheck=False,
    nogil=True,
)
def max_numba(a):
    res = np.zeros((a.shape[0],), dtype=FLOAT_TYPE)
    for i in numba.prange(a.shape[0]):
        res[i] = a[i, :].max()
    return res
    # return np.take_along_axis(array, np.expand_dims(np.argmax(array, axis=1), 1), 1).T


@numba.jit(
    cache=True,
    nopython=True,
    parallel=True,
    fastmath=False,
    boundscheck=False,
    nogil=True,
)
def min_numba(a):
    res = np.zeros((a.shape[0],), dtype=FLOAT_TYPE)
    for i in numba.prange(a.shape[0]):
        res[i] = a[i, :].min()
    return res


# %%
if __name__ == "__main__":
    array = np.array(np.random.sample((10000, 100)), dtype=FLOAT_TYPE)
    # print(fast_arg_top_k(array, 10))

    start = timer()
    a = array.max(axis=1)
    # print(a)
    end = timer()
    print(end - start)

    start = timer()
    c = np.take_along_axis(array, np.expand_dims(np.argmax(array, axis=1), 1), 1).T
    # print(a)
    end = timer()
    print(end - start)

    start = timer()
    b = max_numba(array)
    # print(b)
    end = timer()
    print(end - start)

    print((c == a).all(), (c == b).all(), (a == b).all())
