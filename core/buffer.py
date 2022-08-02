from __future__ import annotations
import os
from enum import Enum
import inspect, importlib, functools, operator
from typing import Union, Tuple, NamedTuple, Any

# These ops will most likely change, but at least get them to work
BinaryOp = Enum("BinaryOp", ["Add", "Mul", "Div", "Pow"])
UnaryOp = Enum("UnaryOp", ["ReLU"])
TensorOp = Enum("TensorOp", ["Matmul"])
LoadOp = Enum("LoadOp", ["fromCpu"])
Ops = Union[BinaryOp, UnaryOp, TensorOp, LoadOp] 

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

def eval_load_op(buf:Buffer):
    assert(buf.op.op == LoadOp.fromCpu)
    return Device.buffers[buf.device].fromCpu(buf.op.arg), [], LoadOp

def eval_unary_op(parents:Buffer):
    real_parents = {x:None for x in parents.op.src}
    for x in real_parents.keys():
        real_parents[x] = x.eval_op(x.device)
    def resolve(x:Union[Buffer, Ops]):
        if isinstance(x, Buffer):
            return real_parents[x]
        if isinstance(x.op, UnaryOp):
            return resolve(x.src[0]).unary_op(x.op)
    return resolve(parents.op), list(real_parents.values()), BinaryOp
    

def eval_tensor_op(parents:Buffer):
    real_parents = {x:None for x in parents.op.src}
    for x in real_parents.keys():
        real_parents[x] = x.eval_op(x.device)
    def resolve(x:Union[Buffer, Ops]):
        if isinstance(x, Buffer):
            return real_parents[x]
        if isinstance(x.op, TensorOp):
            return resolve(x.src[0]).tensor_op(x.op, resolve(x.src[1]))
    return resolve(parents.op), list(real_parents.values()), TensorOp

def eval_binary_op(parents:Buffer):
    real_parents = {x:None for x in parents.op.src}
    for x in real_parents.keys():
        real_parents[x] = x.eval_op(x.device)
    def resolve(x:Union[Buffer, Ops]):
        if isinstance(x, Buffer):
            return real_parents[x]
        if isinstance(x.op, BinaryOp):
            return resolve(x.src[0]).binary_op(x.op, resolve(x.src[1]))
    return resolve(parents.op), list(real_parents.values()), BinaryOp

_eval = {LoadOp: eval_load_op, BinaryOp: eval_binary_op, UnaryOp: eval_unary_op, TensorOp: eval_tensor_op}

class Buffer:
    def __init__(self, op:Ops, op_type, device):
        self.device = device
        self.op = op
        self.op_type = op_type

    def __repr__(self):
        return f"<Buffer op: {self.op.op}  device: {self.device}>"

    @staticmethod
    def fromCpu(x, device):
        return Buffer(op=Ops(LoadOp.fromCpu, tuple(), x.copy()), op_type=LoadOp, device=device)

    def binary_op(x, op, y):
        assert x.device == y.device
        src = tuple(x.op if x.op_type == BinaryOp else i for i in tuple([x, y]))
        buf = Buffer(Ops(op, src), BinaryOp, x.device)
        return eval_binary_op(buf)[0]

    def unary_op(self, x):
        src = tuple(self.op if self.op_type == UnaryOp else i for i in tuple([self]))
        buf = Buffer(Ops(x, src), UnaryOp, self.device)
        return eval_unary_op(buf)[0]
    
    def tensor_op(x, op, y):
        assert x.device == y.device
        src = tuple(x.op if x.op_type == TensorOp else i for i in tuple([x, y]))
        buf = Buffer(Ops(op, src), TensorOp, x.device)
        return eval_tensor_op(buf)[0]

    def eval_op(self, device=None):
        if device is not None:
            assert(device == self.device)
        evaluated, real_parents, real_type = _eval[self.op_type](self)
        return evaluated

class Ops(NamedTuple):
    op : Ops
    src : Tuple[Ops, Buffer]
    arg : Any = None
