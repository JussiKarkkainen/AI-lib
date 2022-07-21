import graph
import numpy as np

class Ops:
    def __init__(self):
        self.value = None
        self.inputs = []
        self.gradient = None

    def __repr__(self):
        return f"Op: name: {self.name}"
        
class Add(Ops):
    add_counter = 0
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"add/num{add_counter}" if name is None else name
        Add.add_counter += 1

    def forward(self, x, y):
        return x + y

    def backward(self, x, y, dout):
        return dout, dout        

class Mul(Ops):
    mul_counter = 0
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"mul/num={mul_counter}" if name is None else name
        Mul.mul_counter += 1

    def forward(self, x, y):
        return x * y

    def backward(self, x, y, dout):
        return x*dout, y*dout

class Div(Ops):
    div_counter = 0
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"div/num={div_counter}" if name is None else name
        Div.div_counter += 1

    def forward(self, x, y):
        return x * y**-1
    
    def backward(self, x, y, dout):
        return dout/y, dout*x/y**2

class Pow(Ops):
    pow_counter = 0
    def __init__(self, x, y, name=None):
        self.inputs = [x, y]
        self.name = f"pow/num={pow_counter}" if name is None else name
        Pow.pow_counter += 1

    def forwardi(self, x, y):
        return x ** y

    def backward(self, x, y, dout):
        return dout*y*x**(y-1), dout*np.log(a)*x**Y

class Matmul(Ops):
    matmul_counter = 0
    def __init__(self, x, y, name=None):
        self.inputs = [x, y]
        self.name = f"matmul/num={matmul_counter}" if name is None else name
        Matmul.matmul_counter += 1

    def forward(self, x, y):
        return x @ y

    def backward(self, x, y, dout):
        return x.T @ dout, y.T @ dout
