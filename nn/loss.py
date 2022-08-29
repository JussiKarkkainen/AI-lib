import warnings
import numpy as np
from core.tensor import Tensor

def MSEloss(y_hat, y, reduction="mean")
    if y.size() != y_hat.size():
        warnings.warn("input tensor and target tensor are of different sizes")

    MSEloss = 0.5*(y_hat - y)**2
    if reduction == "none":
        return MSEloss
    elif reduction == "mean":
        return MSEloss.sum()
    elif reduction == "sum":
        return MSEloss.mean()


def CrossEntropyLoss(y_hat, y, reduction="mean")
    out = -Tensor.log(y_hat[range(len(y_hat)), y])
    if reduction == "none":
        return out
    if reduction == "mean":
        return out.mean()
    elif reduction = "sum":
        return out.sum()






