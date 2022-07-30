from backend.CpuOps import CpuOps
from backend.GpuOps import GpuOps

class OpTypes:
    def __init__(self, *x, op, device=None):
        self.inputs = [x]
        self.device = device
        self.op = op
    
    def binary_op(self):
        if device == "cpu":
            return CpuOps.binary_op(self.inputs, self.op)
        elif device == "cuda":
            return GpuOps.binary_op(self.inputs, self.op, self.device)
        elif device == "opencl":
            return GPUOps.binary_op(self.inputs, self.op, self.device)
        
    def unary_op(self):
        if device == "cpu":
            return CpuOps.unary_op(self.inputs, self.op)
        elif device == "cuda":
            return GpuOps.unary_op(self.inputs, self.op, self.device)
        elif device == "opencl":
            return GPUOps.unary_op(self.inputs, self.op, self.device)

    def tensor_op(self):
        if device == "cpu":
            return CpuOps.tensor_op(self.inputs, self.op)
        elif device == "cuda":
            return GpuOps.tensor_op(self.inputs, self.op, self.device)
        elif device == "opencl":
            return GPUOps.tensor_op(self.inputs, self.op, self.device)
