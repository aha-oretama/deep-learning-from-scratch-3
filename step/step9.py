if '__file__' in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from dezero import Variable
from dezero.functions import square, exp

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)
