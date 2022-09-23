from core.tensor import Tensor


class Context:
    def __init__(self, params, state):
        self._params = params
        self._state = state

    def get_params(self):
        return self._params

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



def transform(f):
    ''' 
    Transforms a function constructed with nn.Module into a
    pair of pure functions, init and apply. Works in the same 
    way as hk.transform in Haiku.
    '''
    def init_fn(x_init, y_init):
        with new_ctx() as ctx:
            try:
                f(x_init, y_init)
            except Exception as e: 
                print(e)
        return ctx.get_params(), ctx.get_state()


    def apply_fn():
        pass

    tie_fn(f, init_fn, apply_fn)

    return Transformed(init_fn, apply_fn)


def tie_fn(f, init_fn, apply_fn):
    pass

