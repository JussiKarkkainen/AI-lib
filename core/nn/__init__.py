from core.tensor import Tensor
from collections import OrderedDict
import warnings
import numpy as np

class MSELoss:
    def __init__(self, reduction="sum"):
        self.reduction = reduction

    def __call__(self, y_hat, y): 
        out = (y_hat - y)**2
        if self.reduction == "none":
            return out
        elif self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()

class CrossEntropyLoss:
    def __init__(self, reduction="mean"):    
        self.reduction = reduction

    def __call__(self, y_hat, y):
        logprobs = y_hat.logsoftmax()
        labels = np.array(y.data).astype(np.int32)
        out = -1 * (logprobs[range(y.shape[0]), labels])
        if self.reduction == "none":
            return out
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()


# Base class for all models
class Module:
    
    __parameters = []
    
    def __init__(self):
        self.training = True
        self._modules = OrderedDict()
        self._parameters = self.__parameters

    def forward(self, x):
        raise NotImplementedError
   
    def parameters(self, recurse=True):
        return self._parameters

    def modules(self):
        for _, module in self.modules:
            yield module
    
    def add_module(self, name, module):
        if not isinstance(module, Module):
            raise TypeError("module needs to be a Module subclass")
        if hasattr(self, name) and name not in self._modules:
            raise AttributeError("module exists")
        self._modules[name] = module
    
    def add_params(self, params):
        for param in params:
            self._parameters.append(param)

    def summarize(self):
        pass

    def __call__(self, *inputs): 
        return self.forward(*inputs) 

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor.randn(in_features, out_features)
        self.bias = Tensor.zeros(out_features) if bias else None
        self._parameters = self.add_params((self.weights, self.bias))
        
    def forward(self, x):
        return x.matmul(self.weights) + self.bias

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = Tensor.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = Tensor.zeros(self.out_channels) if bias else None 
        self._parameters = self.add_params((self.weights, self.bias))

    def forward(self, x):
        return x.conv2d(self.weights, self.bias, padding=self.padding, stride=self.stride)

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return x.maxpool2d(self.kernel_size, self.stride)
