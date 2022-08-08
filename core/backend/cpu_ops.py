import numpy as np
from core.buffer import UnaryOp, BinaryOp, ReduceOp, TransformOp, TensorOp

class CpuBuffer(np.ndarray):

    def mul(x, y):
        return np.multiply(x, y)
    def add(x, y):
        return np.add(x, y)
    def relu(x):
        return np.maximum(x, 0)
    def power(x, y):
        return np.power(x, y)
    def div(x, y):
        return np.divide(x, y)
    def matmul(x, y):
        return np.matmul(x, y)
    def max(x, *args, **kwargs):
        return np.max(x, *args, **kwargs)
    def sum(x, axis, keepdims):
        x = np.asarray(x)
        return np.sum(x, axis=axis, keepdims=keepdims)
    def reshape(x, arg):
        return np.reshape(x, arg)
    def permute(x, arg):
        return np.transpose(x, arg)
    def expand(x, arg):
        return np.broadcast_to(x, arg).view(CpuBuffer)

    @staticmethod
    def fromCpu(x):
        return x.view(CpuBuffer) 
    def toCpu(x):
        return x

    def unary_op(x, op):
        if op == UnaryOp.ReLU:
            return CpuBuffer.relu(x)

    def binary_op(x, op, y):
        if op == BinaryOp.Add:
            return CpuBuffer.add(x, y)
        elif op == BinaryOp.Mul:
            return CpuBuffer.mul(x, y)
        elif op == BinaryOp.Div:
            return CpuBuffer.div(x, y)
        elif op == BinaryOp.Pow:
            return CpuBuffer.power(x, y)

    def reduce_op(x, op, axis):
        if op == ReduceOp.Sum:
            return CpuBuffer.sum(x, axis, keepdims=True)
        elif op == ReduceOp.Max:
            return CpuBuffer.max(x, axis, keepdims=True)

    def transform_op(x, op, arg=None):
        if op == TransformOp.Reshape:
            return x.reshape(arg)
        elif op == TransformOp.Permute:
            return x.permute(arg)
        elif op == TransformOp.Expand:
            return x.expand(arg)

    def tensor_op(x, op, y):
        if op == TensorOp.Matmul:
            return CpuBuffer.matmul(x, y)
