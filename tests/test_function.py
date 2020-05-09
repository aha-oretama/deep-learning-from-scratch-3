import unittest

import numpy as np

from dezero import Variable
from dezero.functions import square, exp


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    return (f(x1).data - f(x0).data) / (2 * eps)


def is_valid_gradient(func):
    x = Variable(np.random.rand(1))
    y = func(x)
    y.backward()
    num_grad = numerical_diff(func, x)
    return np.allclose(x.grad.data, num_grad)


class FuncTest(unittest.TestCase):

    def test_square_gradient(self):
        self.assertTrue(is_valid_gradient(square))

    def test_exp_gradient(self):
        self.assertTrue(is_valid_gradient(exp))

    def test_multi_node_gradient(self):
        def func(x):
            return square(square(x)) + square(square(x))
        self.assertTrue(is_valid_gradient(func))


if __name__ == '__main__':
    unittest.main()
