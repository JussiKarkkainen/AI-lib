from core.tensor import Tensor
from typing import NamedTuple

class FrameStack:
    stack = []

    def add(self, frame):
        self.stack.append(frame)
    
    def remove(self, frame):
        self.stack.remove(frame)

frame_stack = FrameStack()

class Frame:

    module_stack = []
    
    def __init__(self, params, state):
        self.params = params
        self.state = state 
    
    @classmethod
    def create(cls, params, state):
        frame = Frame(params=params, state=state)
        return frame
    
    def module(self):
        return module_stack

def current_name():
    frame = current_frame()
    module_state = frame.module_stack if frame.module_stack else None
    module = module_state.module if module_state is not None else None
    return module.name if module is not None else "--" 

def current_frame():
    return frame_stack.stack[0]

def get_param(name, shape):
    frame = current_frame()
    bundle_name = current_name()
    if bundle_name not in frame.params:
        param = None
    else:
        param = frame.params[bundle_name].get(name)
    
    if param is None:
        if name == "w":
            param = Tensor.randn(*shape)
        elif name == "b":
            param = Tensor.zeros(*shape)
        frame.params[bundle_name+name] = {} 
        frame.params[bundle_name+name][name] = param

    return param

class Context:
    def __init__(self, params, state):
        self._params = params
        self._state = state
        self.frame = None

    def get_params(self):
        return self._params
    def get_init_state(self):
        return self._state
    def get_state(self):
        return self._state

    def __enter__(self):
        self.frame = Frame.create(params=self._params, state=self._state)
        frame_stack.add(self.frame)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        frame_stack.remove(self.frame)
        return exc_type is None

def new_ctx(params=None, state=None):
    if params == None:
        params = dict()
    if state == None:
        state = dict()

    return Context(params, state)

class Transformed:
    def __init__(self, init_fn, apply_fn):
        self.init = init_fn
        self.apply = apply_fn

class TransformedState:
    def __init__(self, init_fn, apply_fn):
        self.init = init_fn
        self.apply = apply_fn

def transform(f):
    ''' 
    Transforms a function constructed with nn.Module into a
    pair of pure functions, init and apply. Works in the same 
    way as hk.transform in Haiku.
    '''
    return without_state(transform_with_state(f))

def without_state(f):
    def init_fn(*args, **kwargs):
        params, state = f.init(*args, **kwargs)
        if state:
            raise ValueError("Can't have state in init")
        return params

    def apply_fn(params, *args, **kwargs):
        if "state" in kwargs:
            raise TypeError("State in kwargs")
        out, state = f.apply(params, {}, *args, **kwargs)
        if state:
            raise ValueError("Can't have state in apply")
        return out

    tie_fn(f, init_fn, apply_fn)

    return Transformed(init_fn=init_fn, apply_fn=apply_fn)

def transform_with_state(f):
    
    def init_fn(*args, **kwargs):
        with new_ctx() as ctx:
            try:
                f(*args, **kwargs)
            except Exception as e: 
                print(e)
        return ctx.get_params(), ctx.get_init_state()

    def apply_fn(params, state, *args, **kwargs):
        with new_ctx(params=params, state=state) as ctx:
            try:
                out = f(*args, **kwargs)
            except Exception as e:
                print(e)
        return out, ctx.get_state()

    tie_fn(f, init_fn, apply_fn)

    return Transformed(init_fn, apply_fn)


def tie_fn(f, init_fn, apply_fn):
    if isinstance(f, Transformed):
        f = getattr(f.init, "_original_fn")
    init_fn._original_fn = f
    apply_fn._original_fn = f

