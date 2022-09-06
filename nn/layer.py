from core.tensor import Tensor
from nn.module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor.randn(in_features, out_features)
        self.bias = Tensor.zeros(out_features) if bias else None
        self._parameters = self.add_params(("Lin, weight:", "Lin, bias"), (self.weights, self.bias))
        
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

    def forward(self, x):
        return x.conv2d(self.weights, self.bias, padding=self.padding, stride=self.stride)

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return x.maxpool2d(self.kernel_size, self.stride)
