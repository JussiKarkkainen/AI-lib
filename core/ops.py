import graph
import numpy as np

class Ops:
    def __init__(self):
        self.value = None
        self.inputs = []
        self.grad = None
        
class Add(Ops):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"add" if name is None else name

    def forward(self, x, y):
        return x + y

    def backward(self, x, y, dout):
        return dout, dout        

class Mul(Ops):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"mul" if name is None else name

    def forward(self, x, y):
        return x * y

    def backward(self, x, y, dout):
        return x*dout, y*dout

class Div(Ops):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"div" if name is None else name

    def forward(self, x, y):
        return x * y**-1
    
    def backward(self, x, y, dout):
        return dout/y, dout*x/y**2

class Pow(Ops):
    def __init__(self, x, y, name=None):
        self.inputs = [x, y]
        self.name = f"pow" if name is None else name

    def forwardi(self, x, y):
        return x ** y

    def backward(self, x, y, dout):
        return dout*y*x**(y-1), dout*np.log(a)*x**Y

class Matmul(Ops):
    def __init__(self, x, y, name=None):
        self.inputs = [x, y]
        self.name = f"matmul" if name is None else name

    def forward(self, x, y):
        return x @ y

    def backward(self, x, y, dout):
        return x.T @ dout, y.T @ dout
