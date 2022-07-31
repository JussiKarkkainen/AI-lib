import os
from enum import Enum
import inspect, importlib

# These ops will most likely change, but at least get them to work
BinaryOp = Enum("BinaryOp", ["Add", "Mul", "Div", "Pow"])
UnaryOp = Enum("UnaryOp", ["ReLU"])
TensorOp = Enum("TensorOp", ["Matmul"])
LoadOp = Enum("LoadOp", ["fromCpu"])

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
    def __init__(self, op, op_type, data, device):
        self.device = device
        self.op = op
        self.op_type = op_type
        self.data = data 

    def __repr__(self):
        return f"<Buffer data: {self.data} op: {self.op}  device: {self.device}>"

    @staticmethod
    def fromCpu(x, device):
        return Buffer(op=LoadOp.fromCpu, op_type=LoadOp, data=x, device=device)


    def binary_op(x, op, y):
        assert(x.device == y.device)
        if x.device == "cpu":
            return CpuBuffer.binary_op(x, y, op)
        elif x.device == "cuda" or "opencl":
            return GpuOps.binary_op(x, y, op, self.device)
        
    def unary_op(self, x, y):
        assert(x.device == y.device)
        if x.device == "cpu":
            return CpuBuffer.unary_op(x, y, op)
        elif x.device == "cuda" or "opencl":
            return GpuOps.unary_op(x, y, op, self.device)

    def tensor_op(self, x, y):
        assert(x.device == y.device)
        if x.device == "cpu":
            return CpuBuffer.tensor_op(x, y, op)
        elif x.device == "cuda" or "opencl":
            return GpuOps.tensor_op(x, y, op, self.device)
