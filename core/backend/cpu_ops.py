import numpy as np
from numpy.lib.stride_tricks import as_strided
from enum import Enum
from typing import Union

BinaryOp = Enum("BinaryOp", ["Add", "Mul", "Div", "Pow", "Sub", "Matmul"])
UnaryOp = Enum("UnaryOp", ["ReLU", "Sign", "Exp", "Log"])
LoadOp = Enum("LoadOp", ["fromCpu"])
ReduceOp = Enum("ReduceOp", ["Sum", "Max"])
TransformOp = Enum("TransformOp", ["Reshape", "Permute", "Expand", "Pool2d"])
Ops = Union[BinaryOp, UnaryOp, ReduceOp, TransformOp, LoadOp] 

UnaryOpDict = {'ReLU': lambda x: np.maximum(x, 0), 'Sign': lambda x: np.sign(x),
        'Exp': lambda x: np.exp(x), 'Log': lambda x: np.log(x)}

BinaryOpDict = {'Add': lambda x, y: np.add(x, y), 'Mul': lambda x, y: np.multiply(x, y),
        'Div': lambda x, y: np.divide(x, y), 'Pow': lambda x, y: np.power(x, y), 
        'Sub': lambda x, y: np.substract(x, y), 'Matmul': lambda x, y: np.matmul(x, y)}

class CpuBuffer(np.ndarray):
    
    @staticmethod
    def fromCpu(x):
        return x.view(CpuBuffer)

    def mul(x, y):
        return np.multiply(x, y)
    def add(x, y):
        return np.add(x, y)
    def relu(x):
        return np.maximum(x, 0)
    def exp(x):
        return np.exp(x)
    def log(x):
        return np.log(x) 
    def power(x, y):
        return np.power(x, y)
    def div(x, y):
        return np.divide(x, y)
    def matmul(x, y):
        return np.matmul(x, y)
    def max(x, axis, keepdims):
        x = np.asarray(x)
        return np.max(x, axis=axis, keepdims=True)
    def sum(x, axis, keepdims):
        x = np.asarray(x)
        return np.sum(x, axis=axis, keepdims=True)
    def reshape(x, arg):
        return np.reshape(x, arg)
    def permute(x, arg):
        return np.transpose(x, arg)
    def strided(x, *args):
        return as_strided(x, args[0][0], args[0][1])
    def expand(x, arg):
        return np.broadcast_to(x, arg).view(CpuBuffer)
    def sign(x):
        return np.sign(x)
    
    def unary_op(x, op):
        return (UnaryOpDict[str(op).split('.')[1]])(x).view(CpuBuffer)
    
    def binary_op(x, op, y):
        return (BinaryOpDict[str(op).split('.')[1]])(x, y).view(CpuBuffer)

    def reduce_op(x, op, axis, keepdims=True):
        axis = tuple([i for i,(a,b) in enumerate(zip(x.shape, axis)) if a != b])
        if op == ReduceOp.Sum:
            return CpuBuffer.sum(x, axis, keepdims).view(CpuBuffer)
        elif op == ReduceOp.Max:
            return CpuBuffer.max(x, axis, keepdims).view(CpuBuffer)

    def transform_op(x, op, arg=None):
        if op == TransformOp.Reshape:
            return x.reshape(arg)
        elif op == TransformOp.Permute:
            return x.transpose(arg)
        elif op == TransformOp.Expand:
            return x.expand(arg)
        if op == TransformOp.Pool2d:
            return CpuBuffer.strided(x, arg)

