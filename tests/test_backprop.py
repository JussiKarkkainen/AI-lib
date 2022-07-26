from core.tensor import Tensor
import numpy as np
import torch

class BackPropTest:
    def __init__(self):
        self.x = np.random.randn(1, 3).astype(np.float32)
        self.u = np.random.randn(3,3).astype(np.float32)
        self.v = np.random.randn(3,3).astype(np.float32)
        self.w = np.random.randn(3,3).astype(np.float32)
        self.m = np.random.randn(1,3).astype(np.float32)


    def test_backprop_simple(self):
        def test_backprop():
            x = Tensor(self.x)
            w = Tensor(self.w)
            m = Tensor(self.m)
            out = x.mul(w)
            out = out.add(m)
            out.backward()
            return out.data, x.grad.data, w.grad.data

        def test_backprop_torch():
            x = torch.Tensor(self.x)
            w = torch.Tensor(self.w)
            m = torch.Tensor(self.m)
            out = x.mul(w)
            out = out.add(m)
            out.backward()
            return out.detach().numpy(), x.grad, w.grad 

        for x, y in zip(test_backprop_own(), test_backprop_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)


    def test_backprop_harder(self):
        def test_backprop_own():
            x = Tensor(self.x)
            w = Tensor(self.w)
            m = Tensor(self.m)
            out = x.matmul(w).relu()
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.data, x.grad.data, w.grad.data
        
        def test_backprop_torch_harder():
            x = torch.tensor(self.x, requires_grad=True)
            w = torch.tensor(self.w, requires_grad=True)
            m = torch.tensor(self.m)
            out = x.matmul(w).relu()
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.detach().numpy(), x.grad, w.grad
        
        for x, y in zip(test_backprop_own(), test_backprop_torch()):
            np.testing.assert_allclose(x, y, atol=1e-5)

    
    def test_backprop_diamond(self):
        def test_backprop_own_diamond():
            u = Tensor(self.u)
            v = Tensor(self.v)
            w = Tensor(self.w)
            x = u.mul(v).relu()
            y = u.mul(w).relu()
            out = x.add(y).mul(y).relu()
            out = out.sum()
            out.backward()
            return out.data, u.grad.data, v.grad.data, w.grad.data

        def test_backprop_torch_diamond():
            u = torch.tensor(self.u, requires_grad=True)
            v = torch.tensor(self.v, requires_grad=True)
            w = torch.tensor(self.w, requires_grad=True)
            x = u.mul(v).relu()
            y = u.mul(w).relu()
            out = x.add(y).mul(y).relu()
            out = out.sum()
            out.backward()
            return out.detach().numpy(), u.grad, v.grad, w.grad

        for x, y in zip(test_backprop_own_diamond(), test_backprop_torch_diamond()):
            np.testing.assert_allclose(x, y, atol=1e-5)
