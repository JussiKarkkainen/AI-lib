import numpy as np
from core.tensor import Tensor

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
        ret = Tensor(func.forward(x[0], x[1]), requires_grad=func.requires_grad)
        if func.requires_grad: 
            ret._graph = func
        return ret

BinaryOp = Enum(


class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        '''
        self.out = np.maximum(0, x)
        return self.out
        '''
        return OpType.unary_op(ReLU, x)

    def backward(self, grad):
        return grad * np.clip(self.out, 0, 1)

class Add(Function):
    def forward(self, x, y):
        a = np.asarray(x.data)
        b = np.asarray(y.data)
        out = np.add(a, b)
        return out

    def backward(self, grad_out):
        return grad_out, grad_out  

class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return OpType.binary_op(Mul, x, y)

    def backward(self, x, y, grad_out):
        return x*grad_out, y*grad_out

class Div(Function):
    def forward(self, x, y):
        #return x * y**-1
        return OpType.binary_op(Div, x, y)

    def backward(self, x, y, dout):
        return dout/y, dout*x/y**2

class Pow(Function):
    def forwardi(self, x, y):
        #return x ** y
        return OpType.binary_op(Pow, x, y)

    def backward(self, x, y, dout):
        return dout*y*x**(y-1), dout*np.log(a)*x**Y

class Matmul(Function):
    def forward(self, x, y):
        save_for_backward(x, y)
        #return x @ y
        return OpType.tensor_op(Matmul, x, y)

    def backward(self, x, y, dout):
        return x.T @ dout, y.T @ dout
