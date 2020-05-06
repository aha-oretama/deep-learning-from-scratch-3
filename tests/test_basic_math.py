import unittest

import numpy as np

from dezero.core_simple import Variable
from dezero.utils import array_equal


class AddTest(unittest.TestCase):

    def test_add(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = x0 + x1
        self.assertEqual(y.data, 5)

    def test_add_grad(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = x0 + x1
        y.backward()
        self.assertEqual(x0.grad, 1)
        self.assertEqual(x1.grad, 1)

    def test_add_grad_with_same_variable(self):
        x = Variable(np.array(3))
        y = x + x
        y.backward()
        self.assertEqual(x.grad, 2)

    def test_mul_variable_variable(self):
        a = Variable(np.array(3))
        b = Variable(np.array(4))
        expect = Variable(np.array(12))
        self.assertTrue(array_equal(a * b, expect))

    def test_mul_ndarray_variable(self):
        a = np.array(3)
        b = Variable(np.array(4))
        expect = Variable(np.array(12))
        self.assertTrue(array_equal(a * b, expect))

    def test_mul_variable_ndarray(self):
        a = Variable(np.array(3))
        b = np.array(4)
        expect = Variable(np.array(12))
        self.assertTrue(array_equal(a * b, expect))

    def test_mul_num_variable(self):
        a = 3
        b = Variable(np.array(4))
        expect = Variable(np.array(12))
        self.assertTrue(array_equal(a * b, expect))

    def test_mul_variable_num(self):
        a = Variable(np.array(3))
        b = 4
        expect = Variable(np.array(12))
        self.assertTrue(array_equal(a * b, expect))

    def test_add(self):
        self.assertTrue(array_equal(
            Variable(np.array(3)) + Variable(np.array(4)),
            Variable(np.array(7))
        ))

    def test_neg(self):
        self.assertTrue(array_equal(
            - Variable(np.array(1)),
            Variable(np.array(-1))
        ))

    def test_sub(self):
        self.assertTrue(array_equal(
            Variable(np.array(4)) - Variable(np.array(3)),
            Variable(np.array(1))
        ))

    def test_rsub(self):
        self.assertTrue(array_equal(
            np.array(4) - Variable(np.array(3)),
            Variable(np.array(1))
        ))

    def test_div(self):
        self.assertTrue(array_equal(
            Variable(np.array(12)) / Variable(np.array(3)),
            Variable(np.array(4))
        ))

if __name__ == '__main__':
    unittest.main()
