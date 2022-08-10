import numpy as np
from typing import Optional
import inspect, importlib, pyclbr
from core.buffer import Buffer, Device
from core.backend.cpu_ops import CpuBuffer

class Tensor:
    def __init__(self, data, device=Device.default, requires_grad=True):
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

        self.grad = None 
        self.requires_grad = requires_grad
        self.device = device

        self._graph = None 

    def __repr__(self):
        return f"<Tensor: data={self.data} Grad={self.grad.bufferdata if self.grad else None}>"
        
    @property
    def dtype(self):
        return np.float32

    @property
    def shape(self):
        return np.shape(self.data)

    def device(self):
        return self.device

# These won't be needed in the future, but i'll leave them incase I change my mind
    def topological_sort(self):
        order = []
        visited_nodes = set()
        def _topo(node):
            visited_nodes.add(node)
            if node._graph:
                [_topo(i) for i in node._graph.parents if i not in visited_nodes]
                order.append(node)
            return order
        return _topo(self)

    def backward(self):
        self.grad = Tensor.ones(self.shape, requires_grad=False)
        visited = set()
        for node in reversed(self.topological_sort()):
            if not any(x.requires_grad for x in node._graph.parents): 
                continue
            #assert (node.grad is not None)
            grads = node._graph.backward(node.grad.bufferdata)
            print(node._graph)
            print(grads)
            grads = [Tensor(g, requires_grad=False) if g is not None else None
                for g in ([grads] if len(node._graph.parents) == 1 else grads)] 
            for ins, grad in zip(node._graph.parents, grads):
                if grad is not None and ins.requires_grad:
                    ins.grad = grad if ins.grad is None else ins.grad+grad

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
