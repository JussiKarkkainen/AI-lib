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



class Buffer:
    def __init__(self, op:Ops, op_type, device):
        self.device = device
        self.op = op
        self.op_type = op_type

    def __repr__(self):
        return f"<Buffer op: {self.op}  device: {self.device}>"

    @staticmethod
    def fromCpu(x, device):
        return Buffer(op=Ops(LoadOp.fromCpu, tuple(), x.copy()), op_type=LoadOp, device=device)

    def eval_binary_op(x, op, y):
        assert(x.device == y.device)
        buf = Buffer(Ops(op, [x, y]), BinaryOp, x.device)
        return _binary_op(buf, op)

    def eval_unary_op(self, x, y):
        assert(x.device == y.device)
        if x.device == "cpu":
            return CpuBuffer.unary_op(x, y, op)
        elif x.device == "cuda" or "opencl":
            return GpuOps.unary_op(x, y, op, self.device)

    def eval_tensor_op(self, x, y):
        assert(x.device == y.device)
        if x.device == "cpu":
            return CpuBuffer.tensor_op(x, y, op)
        elif x.device == "cuda" or "opencl":
            return GpuOps.tensor_op(x, y, op, self.device)

    def eval_load_op(buf:Buffer):
        assert(self.op == LoadOp.fromCpu)
        return Device.buffers[self.device].fromCpu(self.op.arg), [], LoadOp

# Needed to get device buffers
def get_buffers(op):
   pass 

def _binary_op(parents:Buffer, op:Ops):
    real_parents = {x:None for x in parents.op.src}
    for x in real_parents.keys():
        print(x.op_type)
        # x.op.op (maybe x.op_type) is what starts _load_op which in turn returns device buffers
        # Need to find a way to call _load_op from here
        real_parents[x] = x.eval 

def _unary_op(buf:Buffer, op:Ops):
    pass
def _tensor_op(*buf:Buffer, op:Ops):
    pass


class Ops(NamedTuple):
    op : Ops
    src : Tuple[Ops, Buffer]
    arg : Any = None
