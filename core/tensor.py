import numpy as np
from typing import Optional
import inspect, importlib, pyclbr
import functools
import math
from core.buffer import Buffer, Device
from core.backend.cpu_ops import CpuBuffer

class Tensor:
    def __init__(self, data, device=Device.default):
        if isinstance(data, list) or isinstance(data, int) or isinstance(data, float):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, Tensor):
            self.data = np.array(data.data)
        elif isinstance(data, CpuBuffer):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.uint8):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.float32):
            self.data = np.array(data, dtype=np.float32)
        else:
            raise Exception(f"Unable to make tensor from {type(data)}")
        if isinstance(self.data, np.ndarray):
            self.bufferdata = Buffer.fromCpu(self.data.astype(np.float32), device)
            self.data = data

        self.device = device
        self._graph = None 

    def __repr__(self):
        return f"<Tensor {self.data} with shape: {self.shape}>"
    
    def __getitem__(self, key):
        return Tensor(self.data[key])

    @property
    def dtype(self):
        return np.float32

    @property
    def shape(self):
        return np.shape(self.data)

    def device(self):
        return self.device
    
    # detach tensor from graph
    def detach(self):
        return Tensor(self.data)

    # Functions for creating tensors 
    @classmethod 
    def arange(cls, end, start=0, **kwargs):
        return cls(np.arange(start, end).astype(np.float32), **kwargs)
    
    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)
    
    @classmethod
    def ones(cls, shape, **kwargs):
        return cls(np.ones(shape, dtype=np.float32), **kwargs) 
    
    @classmethod
    def rand(cls, *shape, **kwargs):
        return cls(np.random.rand(*shape).astype(np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)
    
    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(*shape, dtype=np.float32), **kwargs)
   
    def __neg__(self):
        return self * -1.
    def __sub__(self, x):
        return self + (-x)
    def __rsub__(self, x):
        return self + (-x)
    def tanh(self):
        return 2. * ((2. * self).sigmoid()) - 1.
        #return ((2*self).exp() - 1) / ((2*self).exp() + 1)
    def sigmoid(self):
        return (1. + (-self).exp()) ** -1.
    def sqrt(self):
        return self.pow(0.5)
    def mean(self, axis=None, keepdim=False):
        x = Tensor.sum(self, axis=axis, keepdims=keepdim)
        return x * (math.prod(x.shape)/math.prod(self.shape))
    def softmax(self, dim=1):
        f = self - self.max(axis=len(self.shape)-1, keepdims=True)
        e = f.exp()
        return e / e.sum(axis=len(self.shape)-1, keepdims=True)
    def logsoftmax(self):
        f = self - self.max(axis=len(self.shape)-1, keepdims=True)
        e = f.exp()
        s = e.sum(axis=len(self.shape)-1, keepdims=True)
        return f - s.log()
    
    def flatten(self, start_dim=1, end_dim=-1):
        flat_axis = list(self.shape[start_dim:end_dim])
        flat_axis.append(self.shape[end_dim])
        flat_shape = (functools.reduce(lambda x, y : x*y, flat_axis))
        return Tensor.reshape(self, (self.shape[0], flat_shape))
    
    def relu(self):
        return Tensor.ReLU(self)
    def exp(self):
        return Tensor.Exp(self)
    def log(self):
        return Tensor.Log(self)
    
    @staticmethod
    def broadcast(x, y):
        '''
        x_shape, y_shape = x.shape, y.shape
        cor_shape = x.shape 
        x_broadcasted, y_broadcasted = x.expand(cor_shape), y.expand(cor_shape)
        '''
        tt = [arg for arg in [x,y] if isinstance(arg, Tensor)][0]  # this is the prototype tensor
        x,y = [Tensor([t]) if not isinstance(t, Tensor) else t for t in [x,y]]
        x,y = [t.reshape([1]*(max(len(x.shape), len(y.shape))-len(t.shape)) + list(t.shape)) for t in [x,y]]
        shape_ret = tuple(max(sx, sy) for sx,sy in zip(x.shape, y.shape))
        return x.expand(shape_ret), y.expand(shape_ret)
        #return x_broadcasted, y_broadcasted

    def add(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        x, y = Tensor.broadcast(self, x)
        return Tensor.Add(x, y)
    def mul(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        x, y = Tensor.broadcast(self, x)
        return Tensor.Mul(x, y)
    def pow(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        x, y = Tensor.broadcast(self, x)
        return Tensor.Pow(x, y)
    
    def div(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        return Tensor.Div(self, x)
    def matmul(self, x):
        return Tensor.Matmul(self, x)
    
    def _reshape_conv(x):
        if len(x.shape) == 2:
            x = x.reshape((1, 1, x.shape[0], x.shape[1]))
        return x
    
    def maxpool2d(self, kernel_size, stride, padding=0):
        self = self._reshape_conv()
        return Tensor.Pool2d(self, kernel_size=kernel_size, stride=stride, padding=padding, pooltype="max")
    
    '''
    def avgpool2d(self, kernel_size, stride=1, padding=0):
        self = self._reshape_conv()
        return Tensor.Pool2d(self, kernel_size=kernel_size, stride=stride, padding=padding, pooltype="avg")
    '''
    
    def conv2d(self, w, padding=0, stride=1):
        # Inputs to Corr2d needs to be of shape: input=DxCxHxW, kernel=NKxCxHKxWK
        self = self._reshape_conv()
        w = w._reshape_conv()
        return Tensor.Corr2d(self, w, padding=padding, stride=stride)
    
    def _reduce(self, fxn, axis=None, keepdims=False):
        if axis is None:
            axis = range(len(self.shape))
        if isinstance(axis, int):
            axis = [axis]
        axis = tuple([x if x >= 0 else x+len(self.shape) for x in axis])
        shape = [self.shape[i] for i in range(len(self.shape)) if i not in axis]
        ret = fxn(self, axis=axis)
        return ret if keepdims else ret.reshape(shape=[1] if shape == [] else shape)
    
    def max(self, axis=None, keepdims=False):
        return self._reduce(Tensor.Max, axis=axis, keepdims=keepdims)
    def sum(self, axis=None, keepdims=False):
        #dims = range(len(self.shape) + 1) if axis == None else [axis]
        #dims = tuple([x if x >= 0 else x+len(self.shape) for x in list(dims)])
        out = self._reduce(Tensor.Sum, axis=axis, keepdims=keepdims)
        return out

    def reshape(self, shape):
        return Tensor.Reshape(self, shape=shape)
    def expand(self, shape):
        return Tensor.Expand(self, shape=shape)
    def transpose(self, dims=None):
        return Tensor.Permute(self, dims=dims)

def register(name, function):
    def attach(*x, **kwargs):
        return function.execute(*x, **kwargs)
    setattr(Tensor, name, attach) 
for name, cls in inspect.getmembers(importlib.import_module("core.ops"), inspect.isclass):
    if name not in ["Function", "Enum", "Buffer", "Tensor"] and not name.endswith("Op"):
        register(name, cls)

def register_op(name, op):
    setattr(Tensor, f"__{name}__", op)
    setattr(Tensor, f"__r{name}__", op)
for name in ["add", "mul", "div", "pow", "matmul"]:
    register_op(name, getattr(Tensor, name))
    register_op("truediv", getattr(Tensor, "div"))
