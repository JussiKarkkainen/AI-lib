from core.tensor import Tensor
from collections import OrderedDict
import warnings
import numpy as np
from core.transform import get_param
from core.nn.module import Module, wrap_method

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

class Linear(Module):
    def __init__(self, out_features, bias=True, name=None):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
    
    @wrap_method
    def __call__(self, x):
        self.in_features = x.shape[1]
        w = get_param("w", (self.in_features, self.out_features))
        ret = x.matmul(w)
        if self.bias:
            b = get_param("b", (self.out_features,))
            ret += b
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
       pass 

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
