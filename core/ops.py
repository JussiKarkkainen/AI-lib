from enum import Enum
import numpy as np
from core.tensor import Tensor
from core.buffer import Buffer
from core.buffer import BinaryOp, UnaryOp, ReduceOp, TransformOp, TensorOp
from utils.misc import argsort, im2col_indices, col2im_indices
from core.backend.cpu_ops import CpuBuffer

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
        tmp = self.saved_inputs[0].unary_op(UnaryOp.Sign).unary_op(UnaryOp.ReLU)
        tmp = Buffer.fromCpu(tmp, device="cpu")
        return tmp.binary_op(BinaryOp.Mul, dout)

class Exp(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOp.Exp)

    def vjp(self, dout):
        x = self.saved_inputs[0]
        tmp = x.unary_op(UnaryOp.Exp)
        tmp = Buffer.fromCpu(tmp, device="cpu")
        return tmp.binary_op(BinaryOp.Mul, dout)

class Log(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return x.unary_op(UnaryOp.Log)

    def vjp(self, dout):
        x = self.saved_inputs[0]
        return x.binary_op(BinaryOp.Mul, -1.).binary_op(BinaryOp.Mul, dout)

# BinaryOp
class Add(Function):
    def forward(self, x, y):
        return x.binary_op(BinaryOp.Add, y)
    
    def vjp(self, dout):
        dout = CpuBuffer.fromCpu(dout.op.arg) 
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
        self.save_for_backward(x, y)
        return x.binary_op(BinaryOp.Div, y)

    def vjp(self, dout):
        b = self.saved_inputs[1]
        a = self.saved_inputs[0]
        y_grad = dout.binary_op(BinaryOp.Div, b) 
        tmp = dout.binary_op(BinaryOp.Mul, a)
        tmp = Buffer.fromCpu(tmp, device="cpu")
        tmp1 = Buffer.fromCpu(np.array(2), device="cpu")
        tmp2 = b.binary_op(BinaryOp.Pow, tmp1)
        tmp2 = Buffer.fromCpu(tmp2, device="cpu")
        x_grad = tmp.binary_op(BinaryOp.Div, tmp2) 
        return y_grad, x_grad

class Pow(Function):
    def forward(self, x, y):
        out = x.binary_op(BinaryOp.Pow, y)
        self.save_for_backward(x, y, out)
        return out

    def vjp(self, dout):
        x, y, powxy = self.saved_inputs
        powxy = Buffer.fromCpu(powxy, device="cpu")
        grad_x, grad_y = None, None
        if self.saved_inputs[0]:
            t = Buffer.fromCpu(powxy.binary_op(BinaryOp.Div, x), device="cpu")
            tmp = y.binary_op(BinaryOp.Mul, t)
            tmp = Buffer.fromCpu(tmp, device="cpu")
            grad_x = dout.binary_op(BinaryOp.Mul, tmp)
        if self.saved_inputs[1]:
            tmp = x.unary_op(UnaryOp.Log)
            tmp = Buffer.fromCpu(tmp, device="cpu")
            tmp = tmp.binary_op(BinaryOp.Mul, powxy) 
            tmp = Buffer.fromCpu(tmp, device="cpu")
            grad_y = dout.binary_op(BinaryOp.Mul, tmp)
        return grad_x, grad_y 


#ReduceOp
class Sum(Function):
    def forward(self, x, axis=None):
        self.shape = x.shape
        if axis == None:
            axis = tuple(x.shape[i] for i in range(len(x.shape)))
        return x.reduce_op(ReduceOp.Sum, axis)
    
    def vjp(self, dout):
        return dout.transform_op(TransformOp.Expand, self.shape)

class Max(Function):
    def forward(self, x, axis=None, keepdims=False):
        out = x.reduce_op(ReduceOp.Max, axis=axis)
        if axis == None:
            axis = tuple(x.shape[i] for i in range(len(x.shape)))
        self.save_for_backward(x, out)
        return out

    def vjp(self, dout):
        x, out = self.saved_inputs
        out = Buffer.fromCpu(out, device="cpu")
        out_expanded = out.transform_op(TransformOp.Expand, x.shape)
        max_index = (x.op.arg == out_expanded)
        tmp = Buffer.fromCpu(np.array(1), device="cpu")
        max_index = Buffer.fromCpu(max_index, device="cpu").binary_op(BinaryOp.Mul, tmp)
        div = max_index.reduce_op(ReduceOp.Sum, dout.shape)
        div = Buffer.fromCpu(div, device="cpu")
        div = div.transform_op(TransformOp.Expand, x.shape)
        ret = max_index.binary_op(BinaryOp.Div, div)
        return ret 

#TransformOp
class Reshape(Function):
    def forward(self, x, shape):
        self.new_shape = x.shape
        return x.transform_op(TransformOp.Reshape, shape)
    
    def vjp(self, dout):
        return dout.transform_op(TransformOp.Reshape, self.new_shape)

class Permute(Function):
    def forward(self, x, dims):
        self.dims = dims
        return x.transform_op(TransformOp.Permute, dims)
    
    def vjp(self, dout):
        return dout.transform_op(TransformOp.Permute, tuple(argsort(self.dims)))

class Expand(Function):
    def forward(self, x, shape):
        self.new_shape = x.shape
        return x.transform_op(TransformOp.Expand, shape)
    
    def vjp(self, dout):
        return dout.reduce_op(ReduceOp.Sum, self.new_shape)

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
        assert kernel_size == stride, "Invalid parameters"
        assert H % kernel_size == 0
        assert W % kernel_size == 0 
        self.x_reshape = x.transform_op(TransformOp.Reshape, 
                    (N, C, H // kernel_size, kernel_size, W // kernel_size, kernel_size))
        out = self.x_reshape.reduce_op(ReduceOp.Max, axis=3, keepdims=False)
        out = Buffer.fromCpu(out, device="cpu")
        out = out.reduce_op(ReduceOp.Max, axis=4, keepdims=False)
        out = Buffer.fromCpu(out, device="cpu")
        self.out = out.transform_op(TransformOp.Reshape, (N, C, out.shape[-1], out.shape[-2])) 
        return self.out

    def vjp(self, dout):
        dx_reshaped = np.zeros_like(self.x_reshape)
        out_newaxis = self.out[:, :, :, np.newaxis, :, np.newaxis]
        mask = self.x_reshape == out_newaxis
        mask = np.array(mask)
        dout_newaxis = dout.op.arg[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis.data, dx_reshaped)
        dout_broadcast = dout_broadcast.astype(np.int32)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(self.x.shape)
        return dx

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
        #out = W_cols.tensor_op(TensorOp.Matmul, self.X_cols).binary_op(BinaryOp.Add, self.b)
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
        #b_grad = dout.unary_op(UnaryOp.Sum, axis=(0, 2, 3))
        return x_grad, w_grad
