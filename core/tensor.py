import numpy as np
from core.autograd.backprop import backward

class Tensor:
    def __init__(self, data, device='cpu', requires_grad=True):
        self.data = data
        self.grad = 0
        self.requires_grad = requires_grad
        slef.device = device

    def __repr__(self):
        return f"<Tensor: data={self.data} Grad={self.grad}>"
        
    @property
    def dtype(self):
        return np.float32

    @property
    def shape(self):
        return np.shape(self.data)

    @property
    def device(self)
        return self.device

    def backward(self, grad=None, create_graph=False):
        ''' Compute the gradient of current tensor'''
        return backward(self, grad, create_graph)

    # Functions for basic ops # 
    def __mul__(self, x):
        out = Tensor(self.data * x.data)
        return out

    def __add__(self, x):
        return Tensor(self.data + x.data)

    def __sub__(self, x):
        out = Tensor(self.data - x.data)
        return out

    def __neg__(self):
        return self.data * (-1)

    def __pow__(self, x):
        return Tensor(self.data ** x) 

    # Functions for creating tensors #
    # Same ones as https://pytorch.org/cppdocs/notes/tensor_creation.html #

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
