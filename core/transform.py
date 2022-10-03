from __future__ import annotations
import collections
import functools
import contextlib
import dataclasses
from core.tensor import Tensor
from typing import NamedTuple, List, Dict, Optional, Any, DefaultDict
import collections 

@dataclasses.dataclass
class Frame:
    params: Dict[str, Tensor]
    is_init: bool = False

    module_counts: Dict[str, int] = dataclasses.field(
            default_factory=lambda: collections.defaultdict(lambda: 0))

    call_stack: list = dataclasses.field(default_factory=list)

    def create_param_path(self, identifier) -> str:
        return '/'.join(['~'] + self.call_stack + [identifier])

    def create_unique_module_name(self, module_name: str) -> str:
        number = self.module_counts[module_name]
        self.module_counts[module_name] += 1
        return f"{module_name}_{number}"

frame_stack = []

def current_frame():
    return frame_stack[-1]

def get_param(name, shape):
    frame = current_frame()
    param_path = frame.create_param_path(name)

    if frame.is_init:
        if name == "w":
            x = Tensor([[1., 3., 5., 7., 9., 2., 4., 6., 8.]])
            frame.params[param_path] = x 
            #frame.params[param_path] = Tensor.randn(*shape)
        elif name == "b":
            frame.params[param_path] = Tensor.zeros(*shape)

    if isinstance(frame.params, tuple):
        frame.params = frame.params[0]
    return frame.params[param_path]

class Transformed(NamedTuple):
    init: Callable
    apply: Callable

def transform(f):

    def init_f(*args, **kwargs):
        frame_stack.append(Frame({}, is_init=True))
        f(*args, **kwargs)
        frame = frame_stack.pop()
        return frame.params

    def apply_f(params, *args, **kwargs):
        frame_stack.append(Frame(params))
        outs = f(*args, **kwargs)
        frame_stack.pop()
        return outs

    return Transformed(init_f, apply_f)

