import numpy as np


def make_symmetric(matrix):
    """ generated with GitHub Copilot April 22nd 2024

    :param matrix:
    :return:
    """
    # Convert the input list of lists to a NumPy array
    arr = np.array(matrix)

    # Transpose the array to get the symmetric counterpart
    arr_transpose = arr.T

    # Add the original array and its transpose
    symmetric_arr = arr + arr_transpose

    return symmetric_arr