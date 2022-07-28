import numpy as np
from core.ops import BinaryOp, UnaryOp, TensorOp


class CpuOps:

    def mul(x):
        return np.mul(x, y)
    def add(x, y):
        return np.add(x, y)
    def relu(x):
        return np.maximum(x, 0)
    def power(x, y):
        return np.power(x, y)
    def div(x, y):
        return np.div(x, y)
    def matmul(x, y):
        return np.matmul(x, y)


    def unary_op(x, op):
        if op == UnaryOp.ReLU:
            return x.relu()

    def binary_op(x, y, op):
        if op == BinaryOp.Add:
            return add(x, y)
        elif op == BinaryOp.Mul:
            return mul(x, y)
        elif op == BinaryOp.Div:
            return div(x, y)
        elif op == BinaryOp.Pow:
            return power(x, y)

    def tensor_op(x, y, op):
        if op == TensorOp.Matmul:
            return matmul(x, y)
