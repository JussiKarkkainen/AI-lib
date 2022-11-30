import numpy as np
from typing import Optional
import inspect, importlib, pyclbr
import functools
import math
from AIlib.backend.cpu_ops import CpuBuffer

class Tensor:
    def __init__(self, data):
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
        self.bufferdata = CpuBuffer.fromCpu(self.data)
        self._graph = None 

    def __repr__(self):
        return f"<Tensor {self.data} with shape: {self.shape}>"
        
    def __getitem__(self, key): 
        return self.data[key]
    def __setitem__(self, key, value):
        self.data[key] = value
    
    @property
    def dtype(self):
        return np.float32

    @property
    def shape(self):
        return np.shape(self.data)
    
    @property
    def ndim(self):
        return self.data.ndim

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
    def normal(cls, mean, std, *shape, **kwargs):
        return cls(np.random.normal(mean, std, shape).astype(np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)
    
    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)
    
    @classmethod
    def ones_like(cls, tensor):
        return cls(np.ones_like(tensor, dtype=np.float32))

    def __neg__(self):
        return self * -1.
    def __sub__(self, x):
        return self + (-x)
    def __rsub__(self, x):
        return self + (-x)
    def tanh(self):
        return 2. * ((2. * self).sigmoid()) - 1.
    def sigmoid(self):
        # Numerically stable sigmoid, taken from cs231n
        pos_mask = (self.data >= 0)
        neg_mask = (self.data < 0)
        z = Tensor.zeros(self.shape)
        z.data[pos_mask] = Tensor.exp(Tensor(-self.data[pos_mask])).data
        z.data[neg_mask] = Tensor.exp(Tensor(self.data[neg_mask])).data
        top = Tensor.ones(self.shape)
        top[neg_mask] = z[neg_mask]
        return Tensor(top / (1 + z))
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
        # there is a bug here somewhere that causes a small deviation in gradients
        f = self - self.max(axis=-1, keepdims=True)
        e = f.exp()
        s = e.sum(axis=-1, keepdims=True)
        return f - s.log()
    
    def cross_entropy(self, target):
        return Tensor.CrossEntropy(self, Tensor(target))

    def flatten(self, start_dim=1, end_dim=-1):
        flat_axis = list(self.shape[start_dim:end_dim])
        flat_axis.append(self.shape[end_dim])
        flat_shape = (functools.reduce(lambda x, y : x*y, flat_axis))
        return Tensor.reshape(self, (self.shape[0], flat_shape))

    def sequential(self, l): 
        return functools.reduce(lambda x, f: f(x), l, self)
    
    def relu(self):
        return Tensor.ReLU(self)
    def exp(self):
        return Tensor.Exp(self)
    def log(self):
        return Tensor.Log(self)
    def sigmoid(self):
        return Tensor.Sigmoid(self)

    def add(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        return Tensor.Add(self, x)
    def mul(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        return Tensor.Mul(self, x)
    def pow(self, x):
        x = Tensor(x) if not isinstance(x, Tensor) else x
        return Tensor.Pow(self, x)
    
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
        return Tensor.MaxPool2d(self, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def avgpool2d(self, kernel_size, stride=1, padding=0):
        self = self._reshape_conv()
        return Tensor.AvgPool2d(self, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def conv2d(self, w, padding=0, stride=1):
        # Inputs to Corr2d needs to be of shape: input=DxCxHxW, kernel=NKxCxHKxWK
        self = self._reshape_conv()
        w = w._reshape_conv()
        return Tensor.Corr2d(self, w, padding=padding, stride=stride)
    
    def max(self, axis=None, keepdims=False):
        return Tensor.Max(self, axis=axis, keepdims=keepdims)
    def sum(self, axis=None, keepdims=False):
        return Tensor.Sum(self, axis=axis, keepdims=keepdims)

    def reshape(self, shape):
        return Tensor.Reshape(self, shape=shape)
    def expand(self, shape):
        return Tensor.Expand(self, shape=shape)
    def transpose(self, dims=None):
        return Tensor.Transpose(self, dims=dims)

def register(name, function):
    def attach(*x, **kwargs):
        return function.execute(*x, **kwargs)
    setattr(Tensor, name, attach) 
for name, cls in inspect.getmembers(importlib.import_module("AIlib.ops"), inspect.isclass):
    if name not in ["Function", "Enum", "Buffer", "Tensor"] and not name.endswith("Op"):
        register(name, cls)

def register_op(name, op):
    setattr(Tensor, f"__{name}__", op)
    setattr(Tensor, f"__r{name}__", op)
for name in ["add", "mul", "div", "pow", "matmul"]:
    register_op(name, getattr(Tensor, name))
    register_op("truediv", getattr(Tensor, "div"))
