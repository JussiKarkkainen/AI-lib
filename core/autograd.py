from core.tensor import Tensor
from utils.misc import change_vars


def topological_sort(root):
    order, vis = list(), set()
    def _topo(root):
        if root not in vis:
            vis.add(root)
            for p in root._graph.parents:
                _topo(p)
                order.append(root)
        return order
    return _topo(root)

def backward(root):
    for node in reversed(topological_sort(root)):
        node.backward()

def grad(func, argnums):
    '''
    Constuct the gradient function that returns the gradient of 
    the given function w.r.t inputs

    func = f(*inputs)
    returns: f'(*inputs) 
    '''
    def gradfun(*args, **kwargs):
        # Replace args with *x
        fun = lambda *x : func(*change_vars(args, argnums, x), **kwargs)
        vjp, ans = make_vjp(fun, args)
        if ans.shape != (1,):
            raise TypeError("Grad only works with scalar output functions")
        return vjp()
    return gradfun

def make_vjp(func, x):
    '''
    Construct function for vector-Jacobian product
    '''
    end_value = func(*x)
    def vjp(): 
        return end_value.backward()
        #return backward(end_value)
    return vjp, end_value
     
