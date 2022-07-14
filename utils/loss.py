import warnings
import numpy as np


class MSEloss:
    def __init(self, reduction="mean"):
        self.reduction = reduction


    def forward(in_tensor, target, reduction="mean"):
        if target.size() != in_tensor.size():
            warnings.warn("input tensor and target tensor are of different sizes")
        diff = []
        for x in range(len(in_tensor)):
            diff.append((in_tensor[x] - target[x])**2)
        if reduction == "none":
            return diff
        elif reduction == "mean":
            return np.mean(diff)
        elif reduction == "sum":
            return np.sum(diff)



class CrossEntropyLoss:
    pass







