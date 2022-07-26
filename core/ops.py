import numpy as np

class Function:
    def __init__(self, tensors, device=None):
        self.parents = tensors
        self.device = device
        self.saved_inputs = [] 
    
    def save_for_backward(self, *x):
        self.saved_inputs.extend(x)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class ReLU(Function):
    def forward(self, x):
        save_for_backward(x)
        self.out = np.maximum(0, x)
        return self.out
    
    def backward(self, grad):
        return grad * np.clip(self.out, 0, 1)

class Add(Function):
    def forward(self, x, y):
        return x + y

    def backward(self, grad_out):
        return grad_out, grad_out  

class Mul(Function):
    def forward(self, x, y):
        save_for_backward(x, y)
        return x * y

    def backward(self, x, y, grad_out):
        return x*grad_out, y*grad_out

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
        save_for_backward(x, y)
        return x @ y

    def backward(self, x, y, dout):
        return x.T @ dout, y.T @ dout
