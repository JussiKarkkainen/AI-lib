from core.tensor import Tensor
from nn.module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor.randn((in_features, out_features))
        self.bias = Tensor.zeros(out_features) if bias else None
        
    def forward(self, x):
        return x.matmul(self.weight) + self.bias

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = Tensor.randn((kernel_size, kernel_size))
        self.bias = Tensor.zeros(kernel_size) if bias else None 

    def forward(self, x):
        return x.conv2d(self.weights, padding=self.padding, stride=self.stride) + b 


class AvgPool2d():
    pass

class MaxPool2d():
    pass
