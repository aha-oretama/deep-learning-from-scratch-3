import unittest

import numpy as np

from dezero import Variable
from dezero.utils import array_equal


class VariableTest(unittest.TestCase):

    def test_repr_none(self):
        self.assertEqual(str(Variable(None)), 'variable(None)')

    def test_repr_1dim(self):
        self.assertEqual(str(Variable(np.array(1))), 'variable(1)')

    def test_repr_3dim(self):
        self.assertEqual(
            str(Variable(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))),
            'variable([[1 2 3]\n          [4 5 6]\n          [7 8 9]])'
        )



    def test_rdiv(self):
        self.assertTrue(array_equal(
            np.array(12) / Variable(np.array(3)),
            Variable(np.array(4))
        ))

    def test_pow(self):
        self.assertTrue(array_equal(
            Variable(np.array(3)) ** 2,
            Variable(np.array(9))
        ))

if __name__ == '__main__':
    unittest.main()
