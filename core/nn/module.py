import core.transform as tf

# Module metaclass will be used later to get rid of @wrap_method
'''
class ModuleMetaClass(type):
    def __new__(name):
        method_names = []
        cls = super(ModuleMetaClass).__new__(name)

        for method_name in method_names:
            method = getattr(cls, name)
            method = wrap_method(name, method)
        return cls
'''
# Base class for all models
class Module:
    
    def __init__(self):
        self._unique_name = tf.current_frame().create_unique_module_name(self.__class__.__name__)



def wrap_method(f):
    
    def wrapped(self, *args, **kwargs):
        module_name = self._unique_name
        call_stack = tf.current_frame().call_stack
        call_stack.append(module_name)
        call_stack.append(f.__name__)
        out = f(self, *args, **kwargs)
        assert call_stack.pop() == f.__name__
        assert call_stack.pop() == module_name
        return out

    return wrapped

