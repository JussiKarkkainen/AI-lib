from enum import Enum
import numpy as np
from core.tensor import Tensor
from core.buffer import Buffer
from core.buffer import BinaryOp, UnaryOp, ReduceOp, TransformOp, TensorOp
from utils.misc import argsort, im2col_indices, col2im_indices

class Function:
    def __init__(self, *tensors, device=None):
        self.parents = tuple([x for x in tensors if type(x) == Tensor])
        self.device = device
        self.saved_inputs = [] 
        
    def save_for_backward(self, *x):
        self.saved_inputs.extend(x)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def vjp(self, *args, **kwargs):
        raise NotImplementedError
    
    @classmethod
    def execute(cls, *x, **kwargs):
        func = cls(*x, x[0].device)
        ret = Tensor(func.forward(*[s.bufferdata for s in x], **kwargs))
        ret._graph = func
        return ret

# UnaryOp
class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOp.ReLU)

    def vjp(self, dout):
        return self.saved_inputs[0].unary_op(UnaryOp.Sign).unary_op(UnaryOp.ReLU).binary_op(BinaryOp.Mul, dout)

class Exp(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOp.Exp)

    def vjp(self, dout):
        x = self.saved_inputs[0]
        return x.unary_op(UnaryOp.Exp).binary_op(BinaryOp.Mul, dout)

# BinaryOp
class Add(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Add, y)

    def vjp(self, dout):
        return dout, dout

class Mul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.binary_op(BinaryOp.Mul, y)

    def vjp(self, dout):
        x_grad = self.saved_inputs[1].binary_op(BinaryOp.Mul, dout)
        y_grad = self.saved_inputs[0].binary_op(BinaryOp.Mul, dout)
        return x_grad, y_grad 

class Div(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Div, y)

    def vjp(self, x, y, dout):
        b = self.saved_inputs[1]
        a = self.saved_inputs[0]
        y_grad = dout.binary_op(BinaryOp.Div, b) 
        x_grad = dout.binary_op(BinaryOp.Mul, a).binary_op(BinaryOp.Div, b.binary_op(BinaryOp.Pow, 2))
        return y_grad, x_grad

class Pow(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Pow, y)

    def vjp(self, dout):
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
    
    def vjp(self, dout):
        return dout.transform_op(TransformOp.Expand, self.shape)

class Max(Function):
    def forward(self, x, axis=None):
        out = x.reduce_op(ReduceOp.Max)
        self.save_for_backward(x, out)
        return out

    def vjp(self, dout):
        x, out = self.saved_inputs
        out_reshape = out.transform_op(TransformOp.Expand, x.shape)
        # return 1 in the location of the maximum value * dout, 0 otherwise         

#TransformOp
class Reshape(Function):
    def forward(self, x, shape):
        self.shape = shape
        return x.transform_op(TransformOp.Reshape, shape)
    
    def vjp(self, dout):
        return dout.transform_op(TransformOp.Reshape, self.shape)

class Permute(Function):
    def forward(self, x, dims):
        self.dims = dims
        return x.transform_op(TransformOp.Permute, dims)
    
    def vjp(self, dout):
        return dout.transform_op(TransformOp.Permute, tuple(argsort(self.dims)))

class Expand(Function):
    def forward(self, x, shape):
        self.shape = shape
        return x.transform_op(TransformOp.Expand, shape)
    
    def vjp(self, dout):
        return dout.reduce_op(ReduceOp.Sum, self.shape)

# TensorOp
class Matmul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.tensor_op(TensorOp.Matmul, y)

    def vjp(self, dout):
        self.shapex = self.saved_inputs[1].op.arg.shape
        self.shapey = self.saved_inputs[0].op.arg.shape
        x_t = self.saved_inputs[1].transform_op(TransformOp.Permute, self.shapex, True)
        x_grad = dout.tensor_op(TensorOp.Matmul, x_t)
        y_t = self.saved_inputs[0].transform_op(TransformOp.Permute, self.shapey, True)
        y_grad = y_t.tensor_op(TensorOp.Matmul, dout)
        return x_grad, y_grad

class Pool2d(Function):
    def forward(self, x, kernel_size, stride, padding, pooltype):
        self.x = x
        self.pooltype = pooltype
        N, C, H, W = x.shape
        assert H == W, "pool only works on square tensors"
        assert kernel_size == stride
        x_reshape = x.transform_op(TransformOp.Reshape, 
                    (N, C, H // kernel_size, kernel_size, W // kernel_size, kernel_size))
        out = x_reshape.reduce_op(ReduceOp.Max, axis=3, keepdims=False)
        out = Buffer.fromCpu(out, device="cpu")
        out = out.reduce_op(ReduceOp.Max, axis=4, keepdims=False)
        out = Buffer.fromCpu(out, device="cpu")
        out = out.transform_op(TransformOp.Reshape, (N, C, out.shape[-1], out.shape[-2])) 
        return out

    def vjp(self, dout):
        # return dout in places of max values in forward pass
        # [[1, 1, 2, 4], 
        #  [5, 6, 7, 8],
        #  [3, 2, 1, 0],
        #  [1, 2, 3, 4]]
        # kernel_size = 2, stride = 1
        # [[6, 7, 8],
        #  [6, 7, 8],
        #  [3, 3, 4]]
        # Backward pass, when dout = all 1s:
        # [[0, 0, 0, 0], 
        #  [0, 2, 2, 2],
        #  [1, 0, 0, 0],
        #  [0, 0, 1, 1]]
        
        if self.pooltype == "max":
            print(self.out.max(axis=(2, 3)).reshape(self.x.shape[0], self.x.shape[1], self.out_h, self.out_w)) 


class Corr2d(Function):
    def forward(self, x, w, padding, stride):
        ''' Convolution on inputs with shapes:
        x -> input = DxCxHxW
        w -> kernel = NKxCxHKxWk
        '''
        self.save_for_backward(x, w)
        N, C, H, W = x.shape
        self.pad, self.stride = padding, stride
        self.n_K, self.c_K, self.h_K, self.w_K = w.shape
        self.X_cols = im2col_indices(x.op.arg, self.h_K, self.w_K, padding, stride)
        W_cols = w.transform_op(TransformOp.Reshape, (self.n_K, -1))
        out_height = int(((H + 2*padding - self.h_K) / stride + 1))
        out_width = int(((W + 2*padding - self.w_K) / stride + 1))
        out = W_cols.tensor_op(TensorOp.Matmul, self.X_cols)
        out = out.reshape((self.n_K, out_height, out_width, N))
        out = out.transform_op(TransformOp.Permute, (3, 0, 1, 2))
        return out
        
    def vjp(self, dout):
        x, w = self.saved_inputs[0], self.saved_inputs[1]
        dout_T = dout.transform_op(TransformOp.Permute, (1, 2, 3, 0))
        dout_reshape = dout_T.transform_op(TransformOp.Reshape, (self.n_K, -1))
        w_grad = dout_reshape.tensor_op(TensorOp.Matmul, (Buffer.fromCpu(self.X_cols, device="cpu").transform_op(TransformOp.Permute, None)))
        w_grad = w_grad.reshape(w.shape)
        w_reshape = w.transform_op(TransformOp.Reshape, (self.n_K, -1))
        x_grad_col = w_reshape.transform_op(TransformOp.Permute, None).tensor_op(TensorOp.Matmul, dout_reshape)
        x_grad = col2im_indices(x_grad_col, x.shape, self.h_K, self.w_K, padding=self.pad, stride=self.stride)
        return x_grad, w_grad
