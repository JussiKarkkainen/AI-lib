import numpy as np
from core.autograd.backprop import backward

class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = 0
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"<Tensor: data={self.data} Grad={self.grad}>"
        

    # Functions for creating tensors #
    # Same ones as https://pytorch.org/cppdocs/notes/tensor_creation.html #


    def backward(self, grad=None, create_graph=False):
        ''' Compute the gradient of current tensor'''
        return backward(self, gradient, create_graph)


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
