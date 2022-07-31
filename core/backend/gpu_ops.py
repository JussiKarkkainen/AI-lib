


class GpuBuffer:

    def __init__(self, *inputs, op, device):
        self.inputs = [inputs]
        self.op = op
        self.device = device

    def __repr__(self):
        return f"<GpuBuffer"

    @staticmethod
    def fromCpu(x):
        pass 

    def unary_op(self):
        pass
    def binary_op(self):
        pass
    def tensor_op(self):
        pass
