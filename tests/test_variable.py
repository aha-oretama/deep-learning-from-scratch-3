import unittest
from core import Variable
import numpy as np


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

if __name__ == '__main__':
    unittest.main()
