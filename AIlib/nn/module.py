import AIlib.transform as tf

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

