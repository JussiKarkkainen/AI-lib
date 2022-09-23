from core.tensor import Tensor


class Context:
    def __init__(self, params, state):
        self._params = params
        self._state = state

    def get_params(self):
        return self._params
    def get_init_state(self):
        return self._state
    def get_state(self):
        return self._state

    def __enter__(self):
        return self    
    def __exit__(self, exc_type, exc_val, exc_tb):
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









