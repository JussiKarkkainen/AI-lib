from core.tensor import Tensor
import numpy as np
import torch
import torch.nn.functional as F
import pytest
import math


class MathTest:

    # Tests for all math ops implemented in Tensor class, but not in ops.py
    def __init__(self):
        a = torch.randn(1, 3)
        b = torch.randn(1, 3)
        c = torch.tensor(1, 10)
        d = Tensor.randn(1, 3)
        e = Tensor.randn(1, 3)
        f = Tensor.randn(1, 10)
    
    def test_tanh(self):
        def test_tanh_own(self):
            return f.tanh()
        def test_tanh_torch(self):
            return F.tanh(c)
    def test_neg(self):
        def test_neg_own(self):
            pass
        def test_neg_torch(self):
            pass
    def test_sub(self):
        def test_sub_own(self):
            pass
        def test_sub_torch(self):
            pass
        def test_sigmoid_own(self):
            pass
        def test_sigmoid_torch(self):
            pass
        def test_sqrt_own(self):
            pass
        def test_sqrt_torch(self):
            pass
        def test_mean_own(self):
            pass
        def test_mean_torch(self):
            pass
        def test_softmax_own(self):
            pass
        def test_softmax_torch(self):
            pass
        def test_flatten_own(self):
            pass
        def test_flatten_torch(self):
            pass
