from enum import Enum
import numpy as np
from core.tensor import Tensor
from core.buffer import Buffer
from core.buffer import BinaryOp, UnaryOp, TensorOp

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
        ret = Tensor(func.forward(*[s.bufferdata for s in x]), requires_grad=func.requires_grad)
        #ret = Tensor(func.forward(x[0].bufferdata, x[1].bufferdata), requires_grad=func.requires_grad)
        if func.requires_grad: 
            ret._graph = func
        return ret

# Inputs to ops should be of type CpuBuffer
class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOp.ReLU)

    def backward(self, grad):
        return grad * np.clip(self.out, 0, 1)

class Add(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Add, y)

    def backward(self, grad_out):
        return grad_out, grad_out  

class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.binary_op(BinaryOp.Mul, y)

    def backward(self, x, y, grad_out):
        return x*grad_out, y*grad_out

class Div(Function):
    def forward(self, x, y):
        #return x * y**-1
        return x.binary_op(BinaryOp.Div, y)

    def backward(self, x, y, dout):
        return dout/y, dout*x/y**2

class Pow(Function):
    def forward(self, x, y):
        #return x ** y
        return x.binary_op(BinaryOp.Pow, y)

    def backward(self, x, y, dout):
        return dout*y*x**(y-1), dout*np.log(a)*x**Y

class Matmul(Function):
    def forward(self, x, y):
        save_for_backward(x, y)
        #return x @ y
        return x.tensor_op(TensorOp.Matmul, y)

    def backward(self, x, y, dout):
        return x.T @ dout, y.T @ dout
