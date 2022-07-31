import os
from core.backend.cpu_ops import CpuBuffer
from core.backend.gpu_ops import GpuBuffer
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
    default : str = [dev for dev in devices if os.getenv(str(dev)) is not None] 


class Buffer:
    def __init__(self, op, op_type, device=None):
        self.device = device
        self.op = op
        self.op_type = op_type
    

    def __repr__(self):
        return f"<Buffer op: {self.op} device: {self.device}>"

    @staticmethod
    def fromCpu(self, device):
        return Buffer(op=LoadOp.fromCpu, op_type=LoadOp, device=device)

    def binary_op(self, x, y):
        if self.device == "cpu":
            return CpuBuffer.binary_op(self.inputs, self.op)
        elif self.device == "cuda":
            return GpuOps.binary_op(self.inputs, self.op, self.device)
        elif self.device == "opencl":
            return GPUOps.binary_op(self.inputs, self.op, self.device)
        
    def unary_op(self, x, y):
        if self.device == "cpu":
            return CpuBuffer.unary_op(self.inputs, self.op)
        elif self.device == "cuda":
            return GpuOps.unary_op(self.inputs, self.op, self.device)
        elif self.device == "opencl":
            return GPUOps.unary_op(self.inputs, self.op, self.device)

    def tensor_op(self, x, y):
        if self.device == "cpu":
            return CpuBuffer.tensor_op(self.inputs, self.op)
        elif self.device == "cuda":
            return GpuOps.tensor_op(self.inputs, self.op, self.device)
        elif self.device == "opencl":
            return GPUOps.tensor_op(self.inputs, self.op, self.device)
