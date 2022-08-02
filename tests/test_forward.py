from core.tensor import Tensor
import numpy as np
import torch
import pytest


class ForwardTest:
    def __init__(self):
        self.x = np.random.randn(1,3).astype(np.float32)
        self.u = np.random.randn(3,3).astype(np.float32)
        self.v = np.random.randn(3,3).astype(np.float32)
        self.w = np.random.randn(3,3).astype(np.float32)
        self.m = np.random.randn(1,3).astype(np.float32)
        self.f = np.random.randn(1,3).astype(np.float32)

    def test_add_1dim(self):
        def test_add_1dim_own(self):
            x = Tensor(self.x)
            m = Tensor(self.m)
            f = Tensor(self.f)
            out = x.add(m)
            out = out.add(f)
            return out.data

        def test_add_1dim_torch(self):
            x = torch.Tensor(self.x)
            m = torch.Tensor(self.m)
            f = torch.Tensor(self.f)
            out = x.add(m)
            out = out.add(f)
            return out.detach().numpy()

        for x, y, in zip(test_add_1dim_own(self), test_add_1dim_torch(self)):
            np.testing.assert_allclose(x, y, atol=1e-5)
            for i, j in zip(x, y):
                assert i == j
 
def test_function():
    a = ForwardTest()
    a.test_add_1dim()


if __name__ == "__main__":
    print("Starting test\n")
    a = ForwardTest()
    a.test_add_1dim()
    print("Finished")
