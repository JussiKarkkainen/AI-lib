import numpy as np
from typing import Optional
import inspect, importlib, pyclbr
import functools
from core.buffer import Buffer, Device
from core.backend.cpu_ops import CpuBuffer

class Tensor:
    def __init__(self, data, device=Device.default):
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, CpuBuffer):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = np.array(data, dtype=np.float32)
        else:
            raise Exception(f"Unable to make tensor from {type(data)}")
        if isinstance(self.data, np.ndarray):
            self.bufferdata = Buffer.fromCpu(self.data.astype(np.float32), device)
            self.data = data

        self.device = device
        self._graph = None 

    def __repr__(self):
        return f"<Tensor: data={self.data}>"
        
    @property
    def dtype(self):
        return np.float32

    @property
    def shape(self):
        return np.shape(self.data)

    def device(self):
        return self.device
    
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
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)
  
    def flatten(self, start_dim=0, end_dim=-1):
        new_shape = list(self.shape[start_dim:end_dim])
        new_shape.append(self.shape[end_dim])
        new_shape = (functools.reduce(lambda x, y : x*y, new_shape),)
        return Tensor.reshape(self, new_shape)

    def relu(self):
        return Tensor.ReLU(self)

    def add(self, x):
        return Tensor.Add(self, x)
    def mul(self, x):
        return Tensor.Mul(self, x)
    def div(self, x):
        return Tensor.Div(self, x)
    def pow(self, x):
        return Tensor.Pow(self, x)
    def matmul(self, x):
        return Tensor.Matmul(self, x)
    
    def conv2d(self, x):
        return Tensor.Conv(self, x)
   
    def sum(self, axis=None):
        axis = range(len(self.shape)) if axis == None else axis
        axis = tuple([x if x >= 0 else x+len(self.shape) for x in axis])
        return Tensor.Sum(self, axis=axis)

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
for name in ["add", "mul", "div", "pow", "matmul"]:
    register_op(name, getattr(Tensor, name))
