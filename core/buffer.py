from __future__ import annotations
import os
from enum import Enum
import inspect, importlib, functools, operator
from typing import Union, Tuple, NamedTuple, Any
import numpy as np

# These ops will most likely change, but at least get them to work
BinaryOp = Enum("BinaryOp", ["Add", "Mul", "Div", "Pow"])
UnaryOp = Enum("UnaryOp", ["ReLU", "Sign", "Exp"])
TensorOp = Enum("TensorOp", ["Matmul", "Conv", "Pool"])
LoadOp = Enum("LoadOp", ["fromCpu"])
ReduceOp = Enum("ReduceOp", ["Sum", "Max"])
TransformOp = Enum("TransformOp", ["Reshape", "Permute", "Expand"])
Ops = Union[BinaryOp, UnaryOp, ReduceOp, TransformOp, TensorOp, LoadOp] 

class Device:
    devices = ["cpu", "gpu"]
    
    def buf_set(devices):
        buffers = {}
        for dev in devices:
            for name, cls in inspect.getmembers(importlib.import_module('core.backend.' + str(dev) + '_ops'), \
                    inspect.isclass):
                if (name.lower() == (dev + "Buffer").lower()):
                    buffers[dev] = cls 
        return buffers
 
    buffers : dict = buf_set(devices)
    default : str = "gpu" if os.getenv("GPU") is not None else "cpu"

# These eval functions can be unified/simplified in the future,
# but for now functionality before optimization

def eval_load_op(buf:Buffer):
    assert(buf.op.op == LoadOp.fromCpu)
    return Device.buffers[buf.device].fromCpu(buf.op.arg), [], LoadOp

def eval_op_all(parents:Buffer, shape=None):
    real_parents = {x:None for x in parents.op.src}
    for x in real_parents.keys():
        real_parents[x] = x.eval_op(x.device)
    def resolve(x:Union[Buffer, Ops]):
        if isinstance(x, Buffer):
            return real_parents[x]
        if isinstance(x.op, UnaryOp):
            return resolve(x.src[0]).unary_op(x.op)
        if isinstance(x.op, BinaryOp):
            return resolve(x.src[0]).binary_op(x.op, resolve(x.src[1]))
        if isinstance(x.op, ReduceOp):
            return resolve(x.src[0]).reduce_op(x.op, shape)
        if isinstance(x.op, TensorOp):
            return resolve(x.src[0]).tensor_op(x.op, resolve(x.src[1]))
        if isinstance(x.op, TransformOp):
            return resolve(x.src[0]).transform_op(x.op, shape)
    return resolve(parents.op), list(real_parents.values()), parents.op_type

class Buffer:
    def __init__(self, op:Ops, op_type, device, shape=None):
        self.device = device
        self.op = op
        self.op_type = op_type
        self.shape = shape

    def __repr__(self):
        return f"<Buffer, shape: {self.shape}  op: {self.op.op} device: {self.device}>"

    @staticmethod
    def fromCpu(x, device):
        return Buffer(op=Ops(LoadOp.fromCpu, tuple(), x.copy()), op_type=LoadOp, device=device, shape=x.shape)

    def binary_op(x, op, y):
        assert x.device == y.device
        src = tuple(x.op if x.op_type == BinaryOp else i for i in tuple([x, y]))
        buf = Buffer(Ops(op, src), BinaryOp, x.device)
        return eval_op_all(buf)[0]

    def unary_op(self, op):
        src = tuple(self.op if self.op_type == UnaryOp else i for i in tuple([self]))
        buf = Buffer(Ops(op, src), UnaryOp, self.device)
        return eval_op_all(buf)[0]
   
    def reduce_op(self, op, axis):
        src = tuple(self.op if self.op_type == ReduceOp else i for i in tuple([self]))
        buf = Buffer(Ops(op, src), ReduceOp, self.device)
        return eval_op_all(buf, axis)[0] 

    def transform_op(self, op, shape, return_buf=False):
        if shape == self.op.arg.shape and (op == TransformOp.Reshape or op == TransformOp.Expand):
            return self.op.arg
        if op == TransformOp.Permute and shape == self.op.arg.shape:
            shape = None
        src = tuple(self.op if self.op_type == TransformOp else i for i in tuple([self]))
        buf = Buffer(Ops(op, src), TransformOp, self.device)
        if return_buf:
            return Buffer.fromCpu(eval_transform_op(buf, shape)[0], self.device)
        return eval_op_all(buf, shape)[0]

    def tensor_op(x, op, y):
        if type(y) == np.ndarray:
            y = Buffer.fromCpu(y, x.device)
        src = tuple(x.op if x.op_type == TensorOp else i for i in tuple([x, y]))
        buf = Buffer(Ops(op, src), TensorOp, x.device)
        return eval_op_all(buf)[0]

    def eval_op(self, device=None):
        if device is not None:
            assert(device == self.device)
        evaluated, real_parents, real_type = eval_load_op(self) if self.op_type == LoadOp else eval_op_all(self)
        return evaluated

class Ops(NamedTuple):
    op : Ops
    src : Tuple[Ops, Buffer]
    arg : Any = None
