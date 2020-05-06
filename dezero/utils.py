import numpy as np

from dezero import Variable


def array_equal(a, b):
    """True if two arrays have the same shape and elements, False otherwise.
    Args:
        a, b (numpy.ndarray or cupy.ndarray or dezero.Variable): input arrays
            to compare
    Returns:
        bool: True if the two arrays are equal.
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    return np.array_equal(a, b)
