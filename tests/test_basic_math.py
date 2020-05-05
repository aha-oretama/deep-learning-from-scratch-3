import unittest
from core import Variable, add
import numpy as np


class AddTest(unittest.TestCase):

    def test_add(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        self.assertEqual(y.data, 5)

    def test_add_grad(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        y.backward()
        self.assertEqual(x0.grad, 1)
        self.assertEqual(x1.grad, 1)

    def test_add_grad_with_same_variable(self):
        x = Variable(np.array(3))
        y = add(x, x)
        y.backward()
        self.assertEqual(x.grad, 2)


if __name__ == '__main__':
    unittest.main()
