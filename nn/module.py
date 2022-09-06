from collections import OrderedDict

# Base class for all models
class Module:
    
    __parameters = {}
    
    def __init__(self):
        self.training = True
        self._modules = OrderedDict()
        self._parameters = self.__parameters

    def forward(self, x):
        raise NotImplementedError
   
    def parameters(self, recurse=True):
        for param in self._parameters.values():
            yield param
    
    def modules(self):
        for _, module in self.modules:
            yield module
    
    def add_module(self, name, module):
        if not isinstance(module, Module):
            raise TypeError("module needs to be a Module subclass")
        if hasattr(self, name) and name not in self._modules:
            raise AttributeError("module exists")
        self._modules[name] = module
    
    def add_params(self, names, params):
        for name, param in zip(names, params):
            self._parameters[name] = param

    def summarize(self):
        pass

    def __call__(self, *inputs): 
        return self.forward(*inputs) 
