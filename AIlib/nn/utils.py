from AIlib.tensor import Tensor
import numpy as np

def one_hot(label, num):
    return Tensor(Tensor.eye(num)[label.data.astype(np.int32)])
