import numpy as np
from core.tensor import Tensor
from enum import Enum
from core.buffer import Buffer


class Function:
    def __init__(self, *tensors, device=None):
        self.parents = tensors
        self.device = device
        self.saved_inputs = [] 
        self.input_grad = [tensor.requires_grad for tensor in self.parents]
        self.requires_grad = any(self.input_grad)


    def save_for_backward(self, *x):
        self.saved_inputs.extend(x)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError
    
    @classmethod
    def execute(cls, *x):
        func = cls(*x)
        ret = Tensor(func.forward(x[0].bufferdata, x[1].bufferdata), requires_grad=func.requires_grad)
        if func.requires_grad: 
            ret._graph = func
        return ret

# Inputs to ops should be of type CpuBuffer
class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        '''
        self.out = np.maximum(0, x)
        return self.out
        '''
        return x.nary_op(ReLU)

    def backward(self, grad):
        return grad * np.clip(self.out, 0, 1)

class Add(Function):
    def forward(self, x, y):
        return x.binary_op(Add, y)

    def backward(self, grad_out):
        return grad_out, grad_out  

class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.binary_op(Mul, y)

    def backward(self, x, y, grad_out):
        return x*grad_out, y*grad_out

class Div(Function):
    def forward(self, x, y):
        #return x * y**-1
        return x.binary_op(Div, y)

    def backward(self, x, y, dout):
        return dout/y, dout*x/y**2

class Pow(Function):
    def forwardi(self, x, y):
        #return x ** y
        return x.binary_op(Pow, y)

    def backward(self, x, y, dout):
        return dout*y*x**(y-1), dout*np.log(a)*x**Y

class Matmul(Function):
    def forward(self, x, y):
        save_for_backward(x, y)
        #return x @ y
        return x.tensor_op(Matmul, y)

    def backward(self, x, y, dout):
        return x.T @ dout, y.T @ dout
