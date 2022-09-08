import warnings
import numpy as np
from core.tensor import Tensor

def MSEloss(y_hat, y, reduction="mean"):
    if y.size() != y_hat.size():
        warnings.warn("input tensor and target tensor are of different sizes")

    MSEloss = 0.5*(y_hat - y)**2
    if reduction == "none":
        return MSEloss
    elif reduction == "mean":
        return MSEloss.sum()
    elif reduction == "sum":
        return MSEloss.mean()


class CrossEntropyLoss:
    
    # Input needs to be a tensor of shape (minibatch, C), C = number of classes or
    # (minibatch, C, d1, d2, d3 ..., dk) with k >= 1 for computing
    # loss per pixel for 2d images

    # Target 

    def __call__(self, y_hat, y, reduction="mean"):
        out = -Tensor.log(Tensor.exp(y_hat) / 
        if reduction == "none":
            return out
        if reduction == "mean":
            return out.mean()
        elif reduction == "sum":
            return out.sum()






