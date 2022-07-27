import numpy as np
from typing import Optional
import inspect, importlib, pyclbr

class Tensor:
    def __init__(self, data, device=None, requires_grad=True):
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.device = device

        # Used for storing computational graph
        self._graph = None


    def __repr__(self):
        return f"<Tensor: data={self.data} Grad={self.grad}>"
        
    @property
    def dtype(self):
        return np.float32

    @property
    def shape(self):
        return np.shape(self.data)

    def device(self):
        return self.device

    
    def topological_sort(self):
        order = []
        visited_nodes = set()
        def _topo(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                if node._graph: 
                    for ctx in node._graph.parents:
                        if ctx not in visited:
                            _topo(ctx)
                order.append(node)
        
            return order


    def backward(self):
        
        self.grad = Tensor.ones(self.shape, requires_grad=False)

        visited = set()
        for node in reversed(self.topological_sort()):
            grads = node._graph.backward(node.grad.data)
            grads = [Tensor(g, requires_grad=False) if g is not None else None
                for g in ([grads] if len(node._graph.parents) == 1 else grads)] 
            for ins, grad in zip(node._graph.parents, grads):
                if ins is not None and grad.requires_grad:
                    ins.grad = grad if ins.grad is None else ins.grad+grad
                    

    # Functions for creating tensors 
    @classmethod 
    def arange(cls, end, start=0, **kwargs):
        return cls(np.arange(start, end).astype(np.float32), **kwargs)
    
    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)
    
    @classmethod
    def ones(cls, *shape, **kwargs):
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
    
    def Add(self, x):
        return Tensor._Add(self, x)
    def Mul(self, x):
        return Tensor._Mul(self, x)
    def Div(self, x):
        return Tensor._Div(Self, x)
    def Pow(self):
        return Tensor._Pow(self, x)
    def Matmul(self, x):
        return Tensor._Matmul(self, x)

def register(name, function):
    def attach(*x):
        return function.execute(*x)
    setattr(Tensor, name if (getattr(Tensor, name, None) is not None) else name, attach) 
for name, cls in inspect.getmembers(importlib.import_module("core.ops"), inspect.isclass):
    if name != "Function":
        register(name, cls)

def register_op(name, op):
    setattr(Tensor, "__{name}__", op)
for op in ["Add", "Mul", "Div", "Pow", "Matmul"]:
    register_op(op, getattr(Tensor, op))
