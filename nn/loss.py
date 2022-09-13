import warnings
import numpy as np
from core.tensor import Tensor

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
        out = -1 * (logprobs[range(y.shape[0]), y.data])

        if self.reduction == "none":
            return out
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()






