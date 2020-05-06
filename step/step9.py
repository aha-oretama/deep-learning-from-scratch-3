import numpy as np

from dezero.core_simple import Variable, square, exp

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)
