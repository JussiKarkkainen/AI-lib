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
    
    @wrap_method
    def __call__(self, x):
        # Input of shape: DxCxHxW
        # Kernel is of shape: NKxCxHKxWK
        w_shape = ((self.out_channels, x.shape[1], self.kernel_size, self.kernel_size))
        b_shape = (self.out_channels, 1) 
        w = get_param("w", w_shape)
        b = get_param("b", b_shape)
        if bias:
            ret = x.conv2d(w, self.padding, self.stride) + b
        else:
            ret = x.conv2d(w, self.padding, self.stride)
        return ret

class BatchNorm2d(Module):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, x):
        return x.batchnorm2d()

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
    
    @wrap_method
    def __call__(self, x):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        dims = tuple([-(i + 1) for i in range(len(self.normalized_shape))])
        mean = x.mean(axis=dims, keepdim=True)
        mean_x2 = (x ** 2).mean(axis=dims, keepdim=True)
        var = mean_x2 - mean ** 2
        gain = get_parameter("b", self.normalized_shape)
        bias = get_parameter("b", self.normalized_shape)
        x_norm = (x - mean) / Tensor.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = gain * x_norm + bias
        return x_norm

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @wrap_method
    def __call__(self, x):
        return x.maxpool2d(self.kernel_size, self.stride, self.padding)

class ScaledDotProductAttention(Module):
    def __init__(self, num_heads, dropout):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heades = num_heads
        self.dropout = dropout
    
    @wrap_method
    def __call__(self, q, k, v):
        d = q.shape[-1]
        out = q.matmul(k.transpose(1, 2)) / Tensor(math_sqrt(d))
        attention = out.softmax(-1)
        out = attention.matmul(v)
        return out, attention

class MultiHeadAttention(Module):
    def __init__(self, num_heads, key_size):
        super().__init__()
        self.num_heads = num_heads
        self.w_q = Linear(num_hiddens, bias=bias)
        self.w_k = Linear(num_hiddens, bias=bias)
        self.w_v = Linear(num_hiddens, bias=bias)
        self.w_o = Linear(num_hiddens, bias=bias)
    
    @wrap_method
    def __call__(self, q, k, v, mask=None):
        pass
