# %%
import numba
import numpy as np
from timeit import default_timer as timer

FLOAT_TYPE = np.float32
FLOAT_BUFFER = np.finfo(FLOAT_TYPE).resolution
# %%


@numba.njit(fastmath=True, parallel=False)
def top_k_numba(array, k):
    """
    Gets the indices of the top k values in an (1-D) array.
    * NOTE: The returned indices are not sorted based on the top values.
    """
    sorted_indices = np.zeros((k,), dtype=FLOAT_TYPE)
    minimum_index = 0
    minimum_index_value = 0
    for value in array:
        if value > minimum_index_value:
            sorted_indices[minimum_index] = value
            minimum_index = sorted_indices.argmin()
            minimum_index_value = sorted_indices[minimum_index]
    return (array >= minimum_index_value).nonzero()[0][::-1][:k]


@numba.njit(fastmath=True, parallel=False)
def min_k_numba(array, k):
    """
    Gets the indices of the min k values in an (1-D) array.
    * NOTE: The returned indices are not sorted based on the top values.
    """
    sorted_indices = np.ones((k,), dtype=FLOAT_TYPE) * 3.4e38
    max_index = 0
    max_value = 3.4e38
    for value in array:
        if value < max_value:
            sorted_indices[max_index] = value
            max_index = sorted_indices.argmax()
            max_value = sorted_indices[max_index]
    return (array <= max_value).nonzero()[0][::-1][:k]


# %%
if __name__ == "__main__":
    array = np.array(np.random.sample((10000,)), dtype=FLOAT_TYPE)
    # print(fast_arg_top_k(array, 10))

    start = timer()
    a = array[np.argpartition(array, -4)[-4:]]
    a.sort()
    print(a)
    end = timer()
    print(end - start)

    start = timer()
    b = array[top_k_numba(array, 4)]
    b.sort()
    print(b)
    end = timer()
    print(end - start)

    start = timer()
    c = array.copy()
    c.sort()
    print(c[-4:])
    end = timer()
    print(end - start)

    print(a == b, b == c[-4:], a == c[-4:])
