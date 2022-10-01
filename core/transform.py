from __future__ import annotations
import collections
import functools
import contextlib
import dataclasses
from core.tensor import Tensor
from typing import NamedTuple, List, Dict, Optional, Any, DefaultDict
import collections 

ModuleState = collections.namedtuple("ModuleState", ("module", "method_name"))
Module = Any
ddict = functools.partial(collections.defaultdict, dict)

class FrameStack:
    def __init__(self):    
        self._stack = collections.deque()

    def add(self, frame):
        self._stack.append(frame)
    
    def pop(self):
        return self._stack.pop()

    def peek(self):
        return self._stack[-1]

    @contextlib.contextmanager
    def __call__(self, elem):
        self.push(elem)
        try:
            yield
        finally:
            assert self.pop() is elem

context_stack = FrameStack()
frame_stack = FrameStack()
current_frame = frame_stack.peek

class Frame(NamedTuple):

    module_stack: FrameStack[ModuleState]
    counter_stack: FrameStack[collections.Counter()]
    used_names_stack: FrameStack[Set[str]]
    params: Dict 
    state: Dict
    
    @classmethod
    def create(cls, params, state):
        frame = Frame(params=params, state=state, module_stack=FrameStack(),
                      counter_stack=FrameStack(), used_names_stack=FrameStack())
        return frame
  
    def assign_name(self, name: str) -> str:
        num = self.module_counts[name]
        self.module_counts[num] += 1
        return f"{name}>{num}"
    

    # Used later with metaclasses, check wrap_method() in module.py in haiku
    @contextlib.contextmanager
    def module(self, module_state: ModuleState):
        with self.module_stack(module_state), \
             self.counter_stack(collections.Counter()), \
             self.used_names_stack(set()):
            yield


def current_name():
    module = current_module()
    return module.name if module is not None else "--" 

def current_module():
    frame = current_frame()
    print(frame.module_stack._stack)
    module_state = frame.module_stack.peek() if frame.module_stack else None
    return module_state.module if module_state is not None else None

def get_param(name, shape):
    frame = current_frame()
    bundle_name = current_name()
    
    full_name = bundle_name + "/" + name
    context = GetContext(full_name=full_name, module=current_module(),
                         shape=shape)
    if bundle_name not in frame.params:
        param = None
    else:
        param = frame.params[bundle_name+name].get(name)
    
    if param is None:
        if name == "w":
            param = Tensor.randn(*shape)
        elif name == "b":
            param = Tensor.zeros(*shape)
        frame.params[bundle_name][name] = param

    return param

class GetContext(NamedTuple):
    full_name: str
    module: Optional[Module]
    shape: tuple()

    @property
    def module_name(self):
        module_name, _ = self.full_name.rsplit("/", 1)
        return module_name

    @property
    def name(self):
        _, name = self.full_name.rsplit("/", 1)
        return name

class Context:
    def __init__(self, params, state):
        self._params = params
        self._state = state
        self._counter = collections.Counter()
        self._names = set()
        self._expected_stack = FrameStack()

    def get_params(self):
        return self._params
    def get_init_state(self):
        return self._state
    def get_state(self):
        return self._state

    def __enter__(self):
        frame = Frame.create(params=self._params, state=self._state)
        frame.used_names_stack.add(self._names)
        frame.counter_stack.add(self._counter)
        self._expected_stack.add(frame)
        context_stack.add(self)
        frame_stack.add(frame)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert frame_stack.pop() is self._expected_stack.pop()
        assert context_stack.pop() is self
        return exc_type is None

def new_ctx(params=None, state=None):
    if params == None:
        params = ddict()
    if state == None:
        state = ddict()

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

