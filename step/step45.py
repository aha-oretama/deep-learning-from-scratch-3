import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
import dezero.layers as L
from dezero import Variable, Model


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size) -> None:
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


np.random.rand(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

hidden_size = 10
model = TwoLayerNet(hidden_size, 1)

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.clear_grads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)

# Plot
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')

t = Variable(np.arange(0, 1, .01)[:, np.newaxis])
y_pred = model(t)
# plt.plot(t.data, y_pred.data, color='r')
# plt.show()
model.plot(t)
