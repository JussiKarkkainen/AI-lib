from collections import OrderedDict


# Base class for all models
class Module:
    def __init__(self):
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()


    def forward(self, x):
        raise NotImplementedError

