import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable, optimizers
from dezero.models import MLP

np.random.rand(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

lr = 0.2
iters = 10000

hidden_size = 10
model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.clear_grads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)

# Plot
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')

t = Variable(np.arange(0, 1, .01)[:, np.newaxis])
y_pred = model(t)
plt.plot(t.data, y_pred.data, color='r')
plt.show()
