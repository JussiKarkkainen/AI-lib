from core.tensor import Tensor


def transform(f):
    ''' 
    Transforms a function constructed with nn.Module into a
    pair of pure functions, init and apply. Works in the same 
    way as hk.transform in Haiku.
    '''
    def init_fn(X_init, y_init):
        pass

    def apply_fn():
        pass

    tie_og_fn(f, init_fn, apply_fn)

    return init_fn, apply_fn

