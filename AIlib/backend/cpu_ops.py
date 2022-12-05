import numpy as np
from numpy.lib.stride_tricks import as_strided
from enum import Enum
from typing import Union

BinaryOp = Enum("BinaryOp", ["Add", "Mul", "Div", "Pow", "Sub", "Matmul"])
UnaryOp = Enum("UnaryOp", ["ReLU", "Sign", "Exp", "Log", "Neg"])
LoadOp = Enum("LoadOp", ["fromCpu"])
ReduceOp = Enum("ReduceOp", ["Sum", "Max"])
TransformOp = Enum("TransformOp", ["Reshape", "Transpose", "Expand", "Strided"])
Ops = Union[BinaryOp, UnaryOp, ReduceOp, TransformOp, LoadOp] 

UnaryOpDict = {'ReLU': lambda x: np.maximum(x, 0), 'Sign': lambda x: np.sign(x),
        'Exp': lambda x: np.exp(x), 'Log': lambda x: np.log(x), 'Neg': lambda x: -x}

BinaryOpDict = {'Add': lambda x, y: np.add(x, y), 'Mul': lambda x, y: np.multiply(x, y),
        'Div': lambda x, y: np.divide(x, y), 'Pow': lambda x, y: np.power(x, y), 
        'Sub': lambda x, y: np.subtract(x, y), 'Matmul': lambda x, y: np.matmul(x, y)}

TransformOpDict = {'Reshape': lambda x, arg: np.reshape(x, arg), 
        'Transpose': lambda x, arg: np.transpose(x, arg), 'Expand': lambda x, arg: np.broadcast_to(x, arg),
        'Strided': lambda x, arg: np.as_strided(x, arg)}

ReduceOpDict = {'Sum': lambda x, axis, keepdims: np.sum(x, axis, keepdims=keepdims),
        'Max': lambda x, axis, keepdims: np.max(x, axis=axis, keepdims=keepdims)}

class CpuBuffer(np.ndarray):
    
    @staticmethod
    def fromCpu(x):
        return x.view(CpuBuffer)
    
    @staticmethod
    def ones_like(x):
        return np.ones_like(x).view(CpuBuffer)
    
    @staticmethod
    def zeros_like(x):
        return np.zeros_like(x).view(CpuBuffer)

    def unary_op(x, op):
        return (UnaryOpDict[str(op).split('.')[1]])(x).view(CpuBuffer)
    
    def binary_op(x, op, y):
        return (BinaryOpDict[str(op).split('.')[1]])(x, y).view(CpuBuffer)
    
    def transform_op(x, op, arg=None):
        return (TransformOpDict[str(op).split('.')[1]])(x, arg).view(CpuBuffer)
    
    def reduce_op(x, op, axis, keepdims=True):
        return (ReduceOpDict[str(op).split('.')[1]])(x, axis, keepdims).view(CpuBuffer)

