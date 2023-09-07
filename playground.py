import numpy as np


def find_index_after_true(binary_array):
    for i, value in enumerate(binary_array):
        if value:
            if all(binary_array[i+1:]):
                return i
    return None


# Sample binary array
binary_array = np.array([False, False, False, True, False, True, True, True, True])

# Starting index
start_index = 3

idx = find_index_after_true(binary_array[start_index:])
if idx is not None:
    print(start_index + idx)
print('AAA')