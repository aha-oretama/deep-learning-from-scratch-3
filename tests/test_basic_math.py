import unittest
from core import Variable, add
import numpy as np


class AddTest(unittest.TestCase):

    def test_add(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        self.assertEqual(y.data, 5)


if __name__ == '__main__':
    unittest.main()
