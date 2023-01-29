from AIlib.tensor import Tensor
from typing import NamedTuple, Any, Callable
import AIlib.nn as nn
import numpy as np

class OptState(NamedTuple):
    params: Any

class Optimizer(NamedTuple):
    init: Callable
    update: Callable

def sgd(lr=0.01):
    def init_fn(params):
        return OptState(params=params)

    def update_fn(grads, state: OptState):
        for k, grad in zip(state.params.keys(), grads.values()):
            state.params[k] -= lr * grad
            # detach here so new params _graph is None
            # TODO: come up with a better solution
            state.params[k] = state.params[k].detach()

        return state.params, OptState(state.params)
         
    return Optimizer(init_fn, update_fn)

