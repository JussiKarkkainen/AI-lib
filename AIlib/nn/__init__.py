from AIlib.tensor import Tensor
from collections import OrderedDict
from typing import NamedTuple, Any
import warnings
import numpy as np
from AIlib.transform import get_param
from AIlib.nn.module import Module, wrap_method
import AIlib.nn.optim as optim

class TrainingState(NamedTuple):
    params: Any
    opt_state: optim.OptState

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

class CategoricalCrossEntropyLoss:
    def __init__(self, reduction="mean"):    
        self.reduction = reduction

    def __call__(self, y_hat, y):
        #out = y_hat.cross_entropy(y)
        out = -Tensor.sum(y*Tensor.logsoftmax(y_hat), axis=-1)
        if self.reduction == "none":
            return out
        elif self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()

class BCELoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, y_hat, y):
        out = Tensor.sum(-y * Tensor.log(y_hat) - (1. - y) * Tensor.log(1. - y_hat))
        if self.reduction == "none":
            return out
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()

class Linear(Module):
    def __init__(self, out_features, bias=True, name=None):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
    
    @wrap_method
    def __call__(self, x):
        self.in_features = x.shape[-1]
        w = get_param("w", (self.in_features, self.out_features))
        b = get_param("b", (self.out_features,))
        ret = x.matmul(w) + b if self.bias else x.matmul(w)
        return ret

class Conv2d(Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, bias=True, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, x):
       ret = x.conv2d(w, b, padding, stride)
       return ret

'''
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
'''

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return x.maxpool2d(self.kernel_size, self.stride)


class ScaledDotProductAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        pass

    def __call__(self, q, k, v, mask=None):
        pass

class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        pass

    def __call__(self, q, k, v):
        pass