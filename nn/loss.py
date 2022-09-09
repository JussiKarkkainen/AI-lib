import warnings
import numpy as np
from core.tensor import Tensor

class MSELoss:
    
    def __call__(self, y_hat, y, reduction="mean"):
        if y.size() != y_hat.size():
            warnings.warn("input tensor and target tensor are of different sizes")

        out = 0.5*(y_hat - y)**2
        if reduction == "none":
            return out
        elif reduction == "mean":
            return out.sum()
        elif reduction == "sum":
            return out.mean()


class CrossEntropyLoss:
    
    # Input needs to be a tensor of shape (minibatch, C), C = number of classes or
    # (minibatch, C, d1, d2, d3 ..., dk) with k >= 1 for computing
    # loss per pixel for 2d images

    # Target 

    def __call__(self, y_hat, y, reduction="mean"):
        # out = -Tensor.sum([Tensor.log()
        out = Tensor([1, 2, 3])
        if reduction == "none":
            return out
        if reduction == "mean":
            return out.mean(axis=1)
        elif reduction == "sum":
            return out.sum()






