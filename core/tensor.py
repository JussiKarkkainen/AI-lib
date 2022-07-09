import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = 0
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"<Tensor: data={self.data} Grad={self.grad}>"
        

    # Functions for creating tensors #
    # Same ones as https://pytorch.org/cppdocs/notes/tensor_creation.html #
    
    @classmethod 
    #def arange(self, start=0, end, step=1, *, out=None, 
     #          device=None, requires_grad=False) -> Tensor:
               
        #self.size = (end-start)/step
    
    @classmethod
    def empty(cls, **kwargs):
        pass
    
    @classmethod
    def eye(cls, **kwargs):
        pass 
    
    @classmethod
    def full(cls, **kwargs):
        pass 
    
    @classmethod
    def linspace(cls, **kwargs):
        pass 
    
    @classmethod
    def logspace(cls, **kwargs):
        pass 
    
    @classmethod
    def ones(cls, **kwargs):
        pass 
    
    @classmethod
    def rand(cls, **kwargs):
        pass
    
    @classmethod
    def randint(cls, **kwargs):
        pass
    
    @classmethod
    def randn(cls, **kwargs):
        pass 
    
    @classmethod
    def randperm(cls, **kwargs):
        pass 
    
    @classmethod
    def zeros(cls, *shape):
        return np.zeros(shape, dtype=np.float32)
    
    













