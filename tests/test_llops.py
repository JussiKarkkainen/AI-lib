from core.tensor import Tensor
import numpy as np
import torch
import torch.nn.functional as F
import pytest
import math

class ForwardTest:
    def __init__(self):
        self.x = np.random.randn(1,3).astype(np.float32)
        self.u = np.random.randn(3,3).astype(np.float32)
        self.v = np.random.randn(3,3).astype(np.float32)
        self.w = np.random.randn(3,3).astype(np.float32)
        self.m = np.random.randn(1,3).astype(np.float32)
        self.f = np.random.randn(1,3).astype(np.float32)
        

    def test_binary_op(self):
        def test_binary_own(self):
            x = Tensor(self.x)
            m = Tensor(self.m)
            f = Tensor(self.f)
            out = x.add(m)
            out = out.add(f)
            return out.data

        def test_binary_torch(self):
            x = torch.Tensor(self.x)
            m = torch.Tensor(self.m)
            f = torch.Tensor(self.f)
            out = x.add(m)
            out = out.add(f)
            return out.detach().numpy()

        for x, y, in zip(test_binary_own(self), test_binary_torch(self)):
            np.testing.assert_allclose(x, y, atol=1e-5)
            for i, j in zip(x, y):
                assert i == j
    
    def test_unary_op(self):
        def test_unary_own(self):
            x = Tensor(self.x)
            f = Tensor(self.f)
            m = Tensor(self.m)
            out1 = x.ReLU()
            out2 = f.ReLU()
            out3 = m.ReLU()
            return out1.data, out2.data, out3.data

        def test_unary_torch(self):
            x = torch.Tensor(self.x)
            f = torch.Tensor(self.f)
            m = torch.Tensor(self.m)
            out1 = F.relu(x)
            out2 = F.relu(f)
            out3 = F.relu(m)
            return out1.detach().numpy(), out2.detach().numpy(), out3.detach().numpy()
    
        for x, y in zip(test_unary_own(self), test_unary_torch(self)):
            np.testing.assert_allclose(x, y, atol=1e-5)
            for i, j in zip(x, y):
                for k, p in zip(i, j):
                    assert math.isclose(k, p, rel_tol=1e-5)
    
    def test_reduce_op(self):
        def test_reduce_own(self):
            pass
        def test_reduce_torch(self):
            pass
    
    def test_transform_op(self):
        def test_transform_own(self):
            pass
        def test_transform_torch(self):
            pass

    def test_tensor_op(self):
        def test_matmul_own(self):
            u = Tensor(self.u)
            v = Tensor(self.v)
            out = u.matmul(v)
            return out.data

        def test_matmul_torch(self):
            u = torch.Tensor(self.u)
            v = torch.Tensor(self.v)
            out = u.matmul(v)
            return out.detach().numpy()

        for x, y, in zip(test_matmul_own(self), test_matmul_torch(self)):
            np.testing.assert_allclose(x, y, atol=1e-5)
            for i, j in zip(x, y):
                assert math.isclose(i, j, rel_tol=1e-5)


    def test_forward_all(self):
        self.test_binary_op()
        self.test_unary_op()
        self.test_tensor_op()
        self.test_transform_op()
        self.test_reduce_op()

def test_binary_op():
    a = ForwardTest()
    a.test_binary_op()

def test_unary_op():
    a = ForwardTest()
    a.test_unary_op()

def test_tensor_op():
    a = ForwardTest()
    a.test_tensor_op()

if __name__ == "__main__":
    print("Starting test\n")
    a = ForwardTest()
    a.test_binary_op()
    #a.test_unary_op()
    a.test_tensor_op()
    print("Finished")
