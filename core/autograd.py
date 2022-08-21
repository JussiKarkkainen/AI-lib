from core.tensor import Tensor
from utils.misc import change_vars

def topological_sort(root):
    order, vis = list(), set()
    def _topo(node):
        if node not in vis:
            vis.add(node)
            if node._graph:
                for p in root._graph.parents:
                    _topo(p)
                    order.append(node)
        return order
    return _topo(root)

def backward(root):
    root.grad = Tensor.ones(root.shape)
    for node in reversed(topological_sort(root)):
        grads = node._graph.derivative(node.grad.bufferdata)
        grads = [grads] if len(node._graph.parents) == 1 else grads
        grads = [Tensor(g) for g in grads if g is not None]
        for ins, grad in zip(node._graph.parents, grads):
            if grad is not None:
                ins.grad = grad if ins.grad is None else ins.grad+grad

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
        return backward(end_value)
    return vjp, end_value
     
