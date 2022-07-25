import numpy as np
from tensor import Tensor

def check_node(function, self, y):
    if isinstance(y, Node):
        return function(self, y)
    raise TypeError("Incompatible type") 


class Node:
    def __init__(self):
        pass
    
    def __add__(self, y):
        return check_node(Add, self, y)

    def __mul__(self, y):
        return check_node(Mul, self, y)

    def __div__(self, y):
        return check_node(Div, self, y)

    def __pow__(self, y):
        return check_node(Pow, self, y)

    def __matmul__(self, y):
        return check_node(Matmul, self, y)

class Graph:
    ''' Class for computational graphs
        _graph is a global variable that describes the graph
    '''
    def __init__(self):
        self.ops = set()
    
        global _graph
        _graph = self
    

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
    relu_counter = 0
    def __init__(self, x, name=None):
        super().__init__()
        self.inputs = [x]
        self.name = f"relu/num{ReLU.relu_counter}" if name is not None else name
        ReLU.relu_counter += 1

    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self, grad):
        return grad * np.clip(self.out, 0, 1)


class Add(Function):
    add_counter = 0
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"add/num{Add.add_counter}" if name is None else name
        Add.add_counter += 1

    def forward(self, x, y):
        return x + y

    def backward(self, x, y, dout):
        return dout, dout        

class Mul(Function):
    mul_counter = 0
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"mul/num={Mul.mul_counter}" if name is None else name
        Mul.mul_counter += 1

    def forward(self, x, y):
        return x * y

    def backward(self, x, y, dout):
        return x*dout, y*dout

class Div(Function):
    div_counter = 0
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"div/num={Div.div_counter}" if name is None else name
        Div.div_counter += 1

    def forward(self, x, y):
        return x * y**-1
    
    def backward(self, x, y, dout):
        return dout/y, dout*x/y**2

class Pow(Function):
    pow_counter = 0
    def __init__(self, x, y, name=None):
        self.inputs = [x, y]
        self.name = f"pow/num={Pow.pow_counter}" if name is None else name
        Pow.pow_counter += 1

    def forwardi(self, x, y):
        return x ** y

    def backward(self, x, y, dout):
        return dout*y*x**(y-1), dout*np.log(a)*x**Y

class Matmul(Function):
    matmul_counter = 0
    def __init__(self, x, y, name=None):
        self.inputs = [x, y]
        self.name = f"matmul/num={Matmul.matmul_counter}" if name is None else name
        Matmul.matmul_counter += 1

    def forward(self, x, y):
        return x @ y

    def backward(self, x, y, dout):
        return x.T @ dout, y.T @ dout
