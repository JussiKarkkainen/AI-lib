from enum import Enum
import numpy as np
from core.tensor import Tensor
from core.buffer import Buffer
from core.buffer import BinaryOp, UnaryOp, ReduceOp, TransformOp, TensorOp
from utils.misc import argsort

class Function:
    def __init__(self, *tensors, device=None):
        self.parents = tuple([x for x in tensors if type(x) == Tensor])
        self.device = device
        self.saved_inputs = [] 
        self.input_grad = [x.requires_grad for x in self.parents] 
        self.requires_grad = any(self.input_grad)

    def save_for_backward(self, *x):
        self.saved_inputs.extend(x)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError
    
    @classmethod
    def execute(cls, *x, **kwargs):
        func = cls(*x, x[0].device)
        ret = Tensor(func.forward(*[s.bufferdata for s in x], **kwargs), requires_grad=func.requires_grad)
        if func.requires_grad: 
            ret._graph = func
        return ret

# UnaryOp
class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOp.ReLU)

    def backward(self, grad):
        return grad * np.clip(self.out, 0, 1)

# BinaryOp
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

#ReduceOp
class Sum(Function):
    def forward(self, x, axis=None):
        self.shape = x.shape 
        return x.reduce_op(ReduceOp.Sum)
    
    def backward(self, dout):
        return dout.transform_op(TransformOp.Expand, self.shape)

class Max(Function):
    def forward(self, x, axis=None):
        out = x.reduce_op(ReduceOp.Max)
        self.save_for_backward(x, out)
        return out

    def backward(self, x, dout):
        pass

#TransformOp
class Reshape(Function):
    def forward(self, x, shape):
        self.shape = shape
        return x.transform_op(TransformOp.Reshape, shape)
    
    def backward(self, dout):
        return dout.transform_op(TransformOp.Reshape, self.shape)

class Permute(Function):
    def forward(self, x, dims):
        self.dims = dims
        return x.transform_op(TransformOp.Permute, dims)
    
    def backward(self, dout):
        return dout.transform_op(TransformOp.Permute, tuple(argsort(self.dims)))

class Expand(Function):
    def forward(self, x, shape):
        self.shape = shape
        return x.transform_op(TransformOp.Expand, shape)
    
    def backward(self, dout):
        return dout.reduce_op(ReduceOp.Sum, self.shape)

# TensorOp
class Matmul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        #return x @ y
        return x.tensor_op(TensorOp.Matmul, y)

    def backward(self, x, y, dout):
        return x.T @ dout, y.T @ dout
