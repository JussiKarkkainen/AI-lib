from AIlib.tensor import Tensor
from collections import OrderedDict
from typing import NamedTuple, Any
import warnings
import math
import numpy as np
from AIlib.transform import get_param
from AIlib.nn.module import Module, wrap_method
import AIlib.nn.optim as optim
from AIlib.nn.utils import one_hot

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
    def __init__(self, out_features, bias=True):
        super().__init__()
        self.out_features = out_features
        self.bias = bias
    
    @wrap_method
    def __call__(self, x):
        self.in_features = x.shape[-1]
        w = get_param("w", (self.in_features, self.out_features))
        if self.bias:
            b = get_param("b", (self.out_features,))
            ret = x.matmul(w) + b
        else:
            ret = x.matmul(w)
        return ret

class Conv2d(Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, bias=True, channel_first=True):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.channel_first = True

    @wrap_method
    def __call__(self, x):
        # Input of shape: DxCxHxW
        # Kernel is of shape: NKxCxHKxWK
        if not self.channel_first:
            # Reshape from DxHxWxC to DxCxHxW
            x = x.reshape((x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
        w_shape = ((self.out_channels, x.shape[1], self.kernel_size, self.kernel_size))
        b_shape = (x.shape[0], self.out_channels, 1, 1) 
        w = get_param("w", w_shape)
        if self.bias:
            b = get_param("b", b_shape)
            ret = x.conv2d(w, self.padding, self.stride) + b
        else:
            ret = x.conv2d(w, self.padding, self.stride)
        return ret

class BatchNorm2d(Module):
    def __init__(self, channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.exp_mean = Tensor.zeros(channels)
        self.exp_var = Tensor.zeros(channels)
    
    @wrap_method
    def __call__(self, x):
        batch_size = x.shape[0]
        assert self.channels == x.shape[1]
        x_tmp = x.reshape((batch_size, self.channels, -1))
        # TODO: Differentiate between training and inference
        if not self.track_running_stats:
            mean = x_tmp.mean((0, 2))
            mean_x2 = (x_tmp ** 2).mean((0, 2))
            var = mean_x2 - mean ** 2
            self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
            self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var
        else:
            mean = self.exp_mean
            var = self.exp_var
        x_norm = (x_tmp - mean.reshape((1, -1, 1))) / Tensor.sqrt(var + self.eps).reshape((1, -1, 1))
        
        if self.affine:
            scale = get_param("s", self.channels)
            shift = get_param("sh", self.channels)
            x_norm = scale.reshape((1, -1, 1)) * x_norm + shift.reshape((1, -1, 1))
        
        return x_norm.reshape(x.shape)

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = np.array([normalized_shape])
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    @wrap_method
    def __call__(self, x):
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]
        dims = tuple([-(i + 1) for i in range(len(self.normalized_shape))])
        mean = x.mean(axis=dims, keepdim=True)
        mean_x2 = (x ** 2).mean(axis=dims, keepdim=True)
        var = mean_x2 - mean ** 2
        gain = get_param("g", *self.normalized_shape)
        bias = get_param("b", *self.normalized_shape)
        x_norm = (x - mean) / Tensor.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = gain * x_norm + bias
        return x_norm

class MaxPool2d:
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def __call__(self, x):
        return x.maxpool2d(self.kernel_size, self.stride, self.padding)

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    @wrap_method
    def __call__(self, x):
        x = utils.one_hot(x, self.num_embeddings)
        w = get_param("w", (self.num_embeddings, self.embedding_dim))
        return x.matmul(w)

class MultiHeadAttention(Module):
    def __init__(self, num_heads, d_model, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.w_q = Linear(d_model, bias=False)
        self.w_k = Linear(d_model, bias=False)
        self.w_v = Linear(d_model)
        self.w_o = Linear(d_model)
        self.scale = 1 / math.sqrt(self.d_k)
        self.dropout = dropout

    @wrap_method
    def __call__(self, q, k, v, mask=None):
        batch_size, seq_len, n_embed = q.shape
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
       
        Q = q.reshape((batch_size, -1, self.num_heads, self.d_k)).transpose((0, 2, 1, 3))
        K = k.reshape((batch_size, -1, self.num_heads, self.d_k)).transpose((0, 2, 1, 3))
        V = v.reshape((batch_size, -1, self.num_heads, self.d_k)).transpose((0, 2, 1, 3))
        
        scores = Q @ K.transpose((0, 1, 3, 2))
        scores *= self.scale
        attn = scores.softmax(dim=-1)
        x = attn.dropout(self.dropout).matmul(V)
        x = x.transpose((0, 2, 1, 3))
        x = x.reshape((batch_size, -1, self.d_model))
        x = self.w_o(x)
        return x
