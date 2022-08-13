from core.tensor import Tensor
from nn.module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = Tensor.zeros(out_features) if bias else Non

    def forward(self, x):
        return x.matmul(self.weight) + self.bias
