import numpy as np
from core.autograd.backprop import backward

class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = 0
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"<Tensor: data={self.data} Grad={self.grad}>"
        

    def backward(self, grad=None, create_graph=False):
        ''' Compute the gradient of current tensor'''
        return backward(self, gradient, create_graph)

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

    def matmul(self,

    def mul(self, x):

    def div(self, x):
        return (self * (x ** -1.0)
    
    def add(self, x):
    
    def sub(self, x):
    
    def pow(self, x):
