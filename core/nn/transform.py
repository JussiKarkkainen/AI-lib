from core.tensor import Tensor


class Context:
    def __init__(self, params, state):
        self._params = params
        self._state = state

    def get_params(self):
        return self._params

    def get_state(self):
        return self._state

def new_ctx(params=None, state=None):
    if params == None:
        params = dict()
    if state == None:
        state = dict()

    return Context(params, state)

def transform(f):
    ''' 
    Transforms a function constructed with nn.Module into a
    pair of pure functions, init and apply. Works in the same 
    way as hk.transform in Haiku.
    '''
    def init_fn(X_init, y_init):
        with new_ctx as ctx:
            try:
                f(*args, **kwargs)
            except: 
                print("error")
        return ctx.get_params(), ctx.get_state()


    def apply_fn():
        pass

    tie_fn(f, init_fn, apply_fn)

    return init_fn, apply_fn


def tie_fn(f, init_fn, apply_fn):
    pass

