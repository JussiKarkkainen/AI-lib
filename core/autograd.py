from core.tensor import Tensor
from utils.misc import change_vars, change_var
import pprint

def topological_sort(root):
    order, vis = list(), set()
    def _topo(node):
        if node not in vis:
            vis.add(node)
            if node._graph:
                for p in node._graph.parents:
                    _topo(p)
                order.append(node)
        return order
    return _topo(root)

def backward(g, root):
    gradients = {root: g}
    for node in reversed(topological_sort(root)):
        outgrads = gradients.pop(node)
        outgrads = Tensor(outgrads) if not isinstance(outgrads, Tensor) else outgrads
        grads = node._graph.vjp(outgrads.bufferdata)
        grads = [grads] if len(node._graph.parents) == 1 else grads
        grads = [Tensor(g) for g in grads if g is not None]
        for p, g in zip(node._graph.parents, grads):
            gradients[p] = g if gradients.get(p) is None else gradients.get(p)+g
    #pprint.pprint(gradients)
    return outgrads
    #return [x for x in reversed(gradients.values())] 

def grad(func, argnums=0, return_val=False):
    '''
    Constuct the gradient function that returns the gradient of 
    the given function w.r.t inputs

    func = f(*inputs)
    returns: f'(*inputs) 
    '''
    def gradfun(*args, **kwargs):
        # Replace args with *x
        fun = lambda *x : func(*change_vars(args, argnums, x), **kwargs)
        vjp, ans = make_vjp(fun, args[argnums])
        return vjp(Tensor.ones(ans.shape))
    return gradfun

def make_vjp(func, x):
    '''
    Construct function for vector-Jacobian product
    '''
    end_value = func(x)
    if end_value.shape != () and end_value.shape != (1, 1):
        raise TypeError("Grad only works with scalar output functions")
    def vjp(g): 
        return backward(g, end_value)
    return vjp, end_value
     
