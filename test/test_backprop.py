from core.Tensor import Tensor
from core.ops import ops
import numpy as np
import torch

class BackPropTest:
    def __init__(self):
        self.x = np.random.randn(1, 3).astype(np.float32)
        self.u = np.random.randn(3,3).astype(np.float32)
        self.v = np.random.randn(3,3).astype(np.float32)
        self.w = np.random.randn(3,3).astype(np.float32)
        self.m = np.random.randn(1,3).astype(np.float32)

    def test_backprop(self):
        def test_backprop_own():
            x = Tensor(self.x)
            w = Tensor(self.w)
            m = Tensor(self.m)
            out = x.matmul(w).relu()
            out = out.logsoftmax()
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.data, x.grad.data, w.grad.data
        
        def test_backprop_torch():
            x = torch.tensor(self.x, requires_grad=True)
            w = torch.tensor(self.w, requires_grad=True)
            m = torch.tensor(self.m)
            out = x.matmul(w).relu()
            out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.mul(m).add(m).sum()
            out.backward()
            return out.detach().numpy(), x.grad, w.grad
        
        for x, y in zip(test_backprop_own(), test_backprop_torch()):
            # This tests whether x and y are the same wihtin given tolerance atol
            np.testing.assert_allclose(x, y, atol=1e-5)

    
    def test_backprop_diamond(self):
        def test_backprop_own_diamond():
            u = Tensor(self.u)
            v = Tensor(self.v)
            w = Tensor(self.w)
            x = u.mul(v).relu()
            y = u.mul(w).relu()
            out = x.add(y).mul(y).relu()
            out = out.logsoftmax()
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
            out = torch.nn.functional.log_softmax(out, dim=1)
            out = out.sum()
            out.backward()
            return out.detach().numpy(), u.grad, v.grad, w.grad

        for x, y in zip(test_backprop_own_diamond(), test_backprop_torch_diamond()):
            np.testing.assert_allclose(x, y, atol=1e-5)



if __name__ == "__main__":
    BackPropTest.test_backprop()
    BackPropTest.test_backprop_diamond()