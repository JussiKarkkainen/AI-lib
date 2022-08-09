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

    def backward(self, dout):
        return self.saved_inputs[0].unary_op(UnaryOp.Sign).unary_op(UnaryOp.ReLU).binary_op(BinaryOp.Mul, dout)

# BinaryOp
class Add(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Add, y)

    def backward(self, dout):
        return dout if self.input_grad[0] else None, \
               dout if self.input_grad[1] else None

class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.binary_op(BinaryOp.Mul, y)

    def backward(self, dout):
        x_grad = self.saved_inputs[1].binary_op(BinaryOps.Mul, dout) if self.input_grad[0] else None
        y_grad = self.saved_inputs[0].binary_op(BinaryOps.Mul, dout) if self.input_grad[1] else None
        return x_grad, y_grad 

class Div(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Div, y)

    def backward(self, x, y, dout):
        b = self.saved_inputs[1]
        a = self.saved_inputs[0]
        y_grad = dout.binary_op(BinaryOp.Div, b) 
        x_grad = dout.binary_op(BinaryOp.Mul, a).binary_op(BinaryOp.Div, b.binary_op(BinaryOp.Pow, 2))
        return y_grad, x_grad

class Pow(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Pow, y)

    def backward(self, dout):
        x, y, powxy = ctx.saved_inputs
        grad_x, grad_y = None, None
        if self.input_grad[0]:
            tmp = y.binary_op(BinaryOps.Mul, powxy.binary_op(BinaryOps.Div, x))
            grad_x = dout.binary_op(BinaryOps.Mul, tmp)
        if self.input_grad[1]:
            tmp = x.unary_op(UnaryOps.Log).binary_op(BinaryOps.Mul, powxy) 
            grad_y = dout.binary_op(BinaryOps.Mul, tmp)
        return grad_x, grad_y 

#ReduceOp
class Sum(Function):
    def forward(self, x, axis=None):
        self.shape = x.op.arg.shape 
        return x.reduce_op(ReduceOp.Sum, axis)
    
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
        return x.tensor_op(TensorOp.Matmul, y)

    def backward(self, dout):
        self.shapex = self.saved_inputs[1].op.arg.shape
        self.shapey = self.saved_inputs[0].op.arg.shape
        x_t = self.saved_inputs[1].transform_op(TransformOp.Permute, self.shapex)
        x_grad = dout.tensor_op(TensorOp.Matmul, x_t)
        y_t = self.saved_inputs[0].transform_op(TransformOp.Permute, self.shapey)
        y_grad = y_t.tensor_op(TensorOp.Matmul, dout)
        return x_grad, y_grad
