import numpy as np
from core.autograd.backprop import backward
from graph import _graph
import ops

class Tensor:
    def __init__(self, data, device=None, creators=None, creator_op=None, requires_grad=True):
        self.data = data
        self.grad = 0
        self.requires_grad = requires_grad
        self.device = device
        self.creators = creators
        self.creator_op = creator_op

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
    
    def topological_sort(end_node=None, graph=_graph):
        reverse_order = []
        visited_nodes = set()
        def _topo(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                if isinstance(node, Ops):
                    for input_node = node.inputs.
                        _topo(input_node)
                ordering.append(node)
        
        if head_node = None:
            for node in graph.ops:
                _topo(node)
        else:
            _topo(head_node)

        return reverse_order


    def backward(self):
        pass


    # Functions for creating tensors #

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
