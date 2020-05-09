import numpy as np

from dezero import Function


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x, = self.inputs
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x, = self.inputs
        gx = exp(x) * gy
        return gx


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x, = self.inputs
        return gy * -sin(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y * y)


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tanh(x):
    return Tanh()(x)
