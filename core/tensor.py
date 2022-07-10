import numpy as np
from core.autograd.backprop import backward

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

    def backward(self, grad=None):
        self.grad = grad

        if self.creator_op == "add":
            self.creators[0].backward(grad)
            self.creators[1].backward(grad)
        if self.creator_op == "mul":
        if self.creator_op == "sub":
        if self.creator_op == "neg":
        if self.creator_op == "pow":


    # Functions for basic ops # 
    def __mul__(self, x):
        return Tensor(self.data * x.data,
                      creators=(self, x),
                      creator_op="mul")

    def __add__(self, x):
        return Tensor(self.data + x.data,
                      creators=(self, x),
                      creator_op="add")

    def __sub__(self, x):
        return Tensor(self.data - x.data,
                      creators=(self, x),
                      creator_op="sub")

    def __neg__(self):
        return Tensor(self.data * (-1),
                      creators=[self],
                      creator_op="neg")

    def __pow__(self, x):
        return Tensor(self.data ** x,
                      creators=[self],
                      creator_op = "pow") 

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
