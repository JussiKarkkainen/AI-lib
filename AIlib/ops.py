from enum import Enum
import numpy as np
from AIlib.tensor import Tensor
from utils.misc import argsort, im2col_indices, col2im_indices, col2im_6d
from typing import Union, Tuple, NamedTuple, Any
from AIlib.backend.cpu_ops import BinaryOp, UnaryOp, ReduceOp, TransformOp 
from AIlib.backend.cpu_ops import CpuBuffer

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
        func = cls(*x)
        ret = Tensor(func.forward(*[s.bufferdata for s in x], **kwargs))
        ret._graph = func
        return ret

# UnaryOp
class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOp.ReLU)

    def vjp(self, dout):
        tmp = self.saved_inputs[0].unary_op(UnaryOp.Sign).unary_op(UnaryOp.ReLU)
        return tmp.binary_op(BinaryOp.Mul, dout)

class Exp(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOp.Exp)

    def vjp(self, dout):
        x = self.saved_inputs[0]
        tmp = x.unary_op(UnaryOp.Exp)
        return tmp.binary_op(BinaryOp.Mul, dout)

class Log(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOp.Log)

    def vjp(self, dout):
        x = self.saved_inputs[0]
        return dout.binary_op(BinaryOp.Div, x.binary_op(BinaryOp.Add, 1e-16))

class Sigmoid(Function):
    def forward(self, x):
        self.save_for_backward(x)
        pos_mask = (x >= 0)
        neg_mask = (x < 0)
        z = CpuBuffer.fromCpu(np.zeros_like(x))
        z[pos_mask] = (-x[pos_mask]).unary_op(UnaryOp.Exp)
        z[neg_mask] = (x[neg_mask]).unary_op(UnaryOp.Exp)
        top = CpuBuffer.fromCpu(np.ones_like(x))
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)

    def vjp(self, dout):
        x = self.saved_inputs[0]
        return dout.binary_op(BinaryOp.Mul, (self.forward(x) * (1 - self.forward(x))))        

# BinaryOp
class Add(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Add, y)
    
    def vjp(self, dout):
        return dout, dout

class Sub(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Add, y)

    def vjp(self, dout):
        return dout, dout.unary_op(UnaryOp.Neg)

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
        self.save_for_backward(x, y)
        return x.binary_op(BinaryOp.Div, y.binary_op(BinaryOp.Add, 1e-16))

    def vjp(self, dout):
        b = self.saved_inputs[1]
        a = self.saved_inputs[0]
        y_grad = dout.binary_op(BinaryOp.Div, b) 
        tmp = dout.binary_op(BinaryOp.Mul, a)
        tmp1 = CpuBuffer.fromCpu(np.array(2.))
        tmp2 = b.binary_op(BinaryOp.Pow, tmp1)
        x_grad = tmp.binary_op(BinaryOp.Div, tmp2) 
        return y_grad, x_grad

class Pow(Function):
    def forward(self, x, y):
        out = x.binary_op(BinaryOp.Pow, y)
        self.save_for_backward(x, y, out)
        return out

    def vjp(self, dout):
        x, y, out = self.saved_inputs
        grad_x, grad_y = None, None
        if self.saved_inputs[0].any():
            t = out.binary_op(BinaryOp.Div, x.binary_op(BinaryOp.Add, 1e-16))
            tmp = y.binary_op(BinaryOp.Mul, t)
            grad_x = dout.binary_op(BinaryOp.Mul, tmp)
        if self.saved_inputs[1].any():
            tmp = x.unary_op(UnaryOp.Log)
            tmp = tmp.binary_op(BinaryOp.Mul, out) 
            grad_y = dout.binary_op(BinaryOp.Mul, tmp)
        return grad_x, grad_y 

#ReduceOp
class Sum(Function):
    def forward(self, x, axis=None, keepdims=False):
        self.shape = x.shape
        self.x = x
        return x.reduce_op(ReduceOp.Sum, axis, keepdims=keepdims)
    
    def vjp(self, dout):
        if dout.ndim == 1: dout = dout.transform_op(TransformOp.Reshape, (dout.shape[0], 1))
        if dout.ndim == 2 and self.x.ndim == 3: dout = dout.transform_op(TransformOp.Reshape, (dout.shape[0], dout.shape[1], 1))
        return dout.binary_op(BinaryOp.Mul, CpuBuffer.ones_like(self.x))

class Max(Function):
    def forward(self, x, axis=None, keepdims=False):
        out = x.reduce_op(ReduceOp.Max, axis=axis, keepdims=keepdims)
        self.save_for_backward(x, out)
        return out

    def vjp(self, dout):
        x, out = self.saved_inputs
        out_expanded = out.transform_op(TransformOp.Expand, x.shape)
        max_index = (x == out_expanded)
        tmp = CpuBuffer.fromCpu(np.array(1.))
        max_index = max_index.binary_op(BinaryOp.Mul, tmp)
        div = max_index.reduce_op(ReduceOp.Sum, 0)
        div = div.transform_op(TransformOp.Expand, x.shape)
        ret = max_index.binary_op(BinaryOp.Div, div.binary_op(BinaryOp.Add, 1e-16))
        return ret 

#TransformOp
class Reshape(Function):
    def forward(self, x, shape):
        self.new_shape = x.shape
        return x.transform_op(TransformOp.Reshape, shape)
    
    def vjp(self, dout):
        return dout.transform_op(TransformOp.Reshape, self.new_shape)

class Transpose(Function):
    def forward(self, x, dims):
        self.dims = dims
        return x.transform_op(TransformOp.Transpose, dims)
    
    def vjp(self, dout):
        return dout.transform_op(TransformOp.Transpose, tuple(argsort(self.dims)))

class Expand(Function):
    def forward(self, x, shape):
        self.new_shape = x.shape
        return x.transform_op(TransformOp.Expand, shape)
    
    def vjp(self, dout):
        a = dout.reduce_op(ReduceOp.Sum, self.new_shape)
        return dout.reduce_op(ReduceOp.Sum, self.new_shape)

class Matmul(Function):
    def forward(self, x, y):
        self.save_for_backward(x, y)
        return x.binary_op(BinaryOp.Matmul, y)

    def vjp(self, dout):
        x, y = self.saved_inputs[1], self.saved_inputs[0]
        shapex = x.shape
        shapey = y.shape
        if x.ndim == 3:
            x_t = x.transform_op(TransformOp.Transpose, (0, 2, 1))
        elif x.ndim == 4:
            x_t = x.transform_op(TransformOp.Transpose, (0, 1, 3, 2))
        else:
            x_t = x.transform_op(TransformOp.Transpose)
        x_grad = dout.binary_op(BinaryOp.Matmul, x_t)
        if y.ndim == 3:
            y_t = y.transform_op(TransformOp.Transpose, (0, 2, 1))
        elif x.ndim == 4:
            y_t = y.transform_op(TransformOp.Transpose, (0, 1, 3, 2))
        else:
            y_t = y.transform_op(TransformOp.Transpose)
        y_grad = y_t.binary_op(BinaryOp.Matmul, dout)
        return x_grad, y_grad

class MaxPool2d(Function):
    def forward(self, x, kernel_size, stride, padding):
        N, C, H, W = x.shape
        self.x = x
        pool_height, pool_width = kernel_size, kernel_size 
        self.kernel_size = kernel_size
        self.stride = stride
        assert (H - pool_height) % stride == 0, "Invalid height"
        assert (W - pool_width) % stride == 0, "Invalid width"

        out_height = (H - pool_height) // stride + 1
        out_width = (W - pool_width) // stride + 1

        x_split = x.transform_op(TransformOp.Reshape, (N * C, 1, H, W))
        x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
        x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
        x_cols_max = CpuBuffer.fromCpu(x_cols_max)
        out = x_cols_max.transform_op(TransformOp.Reshape, (out_height, out_width, N, C))
        out = out.transform_op(TransformOp.Transpose, (2, 3, 0, 1))

        self.x_cols = x_cols
        self.x_cols_argmax = x_cols_argmax
        return out
    
    def vjp(self, dout):
        x, x_cols, x_cols_argmax = self.x, self.x_cols, self.x_cols_argmax,
        N, C, H, W = x.shape
        pool_height, pool_width = self.kernel_size, self.kernel_size 
        stride = self.stride 

        dout_reshaped = dout.transform_op(TransformOp.Transpose, (2, 3, 0, 1)).flatten()
        dx_cols = CpuBuffer.zeros_like(x_cols)
        dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
        dx = col2im_indices(
             dx_cols, (N * C, 1, H, W), pool_height, pool_width, padding=0, stride=stride
        )
        dx = CpuBuffer.fromCpu(dx)
        dx = dx.transform_op(TransformOp.Reshape, x.shape)

        return dx

class Corr2d(Function):
    def forward(self, x, w, padding, stride):
        self.x, self.w, self.padding, self.stride = x, w, padding, stride
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
        H += 2 * padding
        W += 2 * padding
        out_h = (H - HH) // stride + 1
        out_w = (W - WW) // stride + 1
        shape = (C, HH, WW, N, out_h, out_w)
        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = x.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        self.x_cols = x_cols
        x_cols.shape = (C * HH * WW, N * out_h * out_w)
        res = w.transform_op(TransformOp.Reshape, (F, -1))
        res = res.binary_op(BinaryOp.Matmul, x_cols)
        res.shape = (F, N, out_h, out_w)
        out = res.transform_op(TransformOp.Transpose, (1, 0, 2, 3))
        out = np.ascontiguousarray(out)
        return out

    def vjp(self, dout):
        x = self.x
        w = self.w
        x_cols = self.x_cols
        stride, pad = self.stride, self.padding 
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, out_h, out_w = dout.shape
        dout_reshaped = dout.transform_op(TransformOp.Transpose, (1, 0, 2, 3))
        dout_reshaped = dout_reshaped.transform_op(TransformOp.Reshape, (F, -1))
        dw = dout_reshaped.binary_op(BinaryOp.Matmul, x_cols.T)
        dw = dw.transform_op(TransformOp.Reshape, w.shape)
        dx_cols = w.transform_op(TransformOp.Reshape, (F, -1))
        dx_cols = dx_cols.transform_op(TransformOp.Transpose)
        dx_cols = dx_cols.binary_op(BinaryOp.Matmul, dout_reshaped)
        dx_cols.shape = (C, HH, WW, N, out_h, out_w)
        dx = col2im_6d(dx_cols, N, C, H, W, HH, WW, pad, stride)
        return dx, dw




