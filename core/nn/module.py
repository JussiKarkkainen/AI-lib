import core.transform as tf

# Base class for all models
class Module:
    
    def __init__(self):
        self._name = tf.current_frame.create_name(self.__class__.__name__)


def wrap_method(f):
    frame = tf.current_frame()
    state = tf.ModuleState(module=self, module_name=self._name)
    
    def wrapped(self, *args, **kwargs):
        with frame.module(state):
            module_name = self._name
            out = f(self, *args, **kwargs)

    return wrapped

