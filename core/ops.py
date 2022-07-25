import numpy as np
from tensor import Tensor

class Function:
    def __init__(self, name='Operator'):
        _graph.ops.add(self)
        self.value = None
        self.inputs = []
        self.gradient = None
    
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __repr__(self):
        return f"Op: name: {self.name}"

class ReLU(Function):
    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self, grad):
        return grad * np.clip(self.out, 0, 1)


class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, x, y, dout):
        return dout, dout        

class Mul(Function):
    def forward(self, x, y):
        return x * y

    def backward(self, x, y, dout):
        return x*dout, y*dout

class Div(Function):
    def forward(self, x, y):
        return x * y**-1
    
    def backward(self, x, y, dout):
        return dout/y, dout*x/y**2

class Pow(Function):
    def forwardi(self, x, y):
        return x ** y

    def backward(self, x, y, dout):
        return dout*y*x**(y-1), dout*np.log(a)*x**Y

class Matmul(Function):
    def forward(self, x, y):
        return x @ y

    def backward(self, x, y, dout):
        return x.T @ dout, y.T @ dout
