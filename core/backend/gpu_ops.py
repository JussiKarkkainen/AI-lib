


class GpuOps:

    def __init__(self, *inputs, op, device):
        self.inputs = [inputs]
        self.op = op
        self.device = device
