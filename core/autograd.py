from core.tensor import Tensor
from utils.misc import change_vars
import pprint
import numpy as np

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

def backward(g, root, input_nodes):
    gradients = {root: g}
    for node in reversed(topological_sort(root)):
        outgrads = gradients.pop(node)
        outgrads = Tensor(outgrads) if not isinstance(outgrads, Tensor) else outgrads
        grads = node._graph.vjp(outgrads.bufferdata)
        grads = [grads] if len(node._graph.parents) == 1 else grads
        grads = [Tensor(g) for g in grads if g is not None]
        for p, g in zip(node._graph.parents, grads):
            g = unbroadcast_grad(p, g)
            assert p.shape == g.shape
            gradients[p] = g if gradients.get(p) is None else gradients.get(p)+g

    if isinstance(input_nodes, dict): 
        ret = dict(input_nodes)
        for k, v in ret.items():
            ret[k] = gradients[v]
    else:
        ret = input_nodes if isinstance(input_nodes, list) else list(input_nodes)
        return [gradients[v] for v in ret]
    
    del gradients
    return ret

def unbroadcast_grad(node, grad):
    correct_grad=grad
    if node.shape != grad.shape:
        dim_diff = np.abs(grad.ndim - node.ndim)
        if dim_diff != 0:
            sum_dims = tuple(range(dim_diff))
            correct_grad = Tensor.sum(grad, axis=sum_dims)
            ones = tuple([axis for axis, size in enumerate(node.shape) if size == 1])
            if len(ones) != 0:
                correct_grad = np.sum(correct_grad, axis=axis, keepdims=True)
        else:
            for i, (g, p) in enumerate(zip(grad.shape, node.shape)):
                if g != p:
                    g_sum = Tensor.sum(grad, axis=i)
                    # To account for (1,), () and others
                    correct_grad = g_sum.reshape(p)
    return correct_grad

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
        vjp, ans = make_vjp(fun, tuple([args[([argnums] if isinstance(argnums, int) else argnums)[i]] \
            for i in ([argnums] if isinstance(argnums, int) else argnums)])) 
        return vjp(Tensor.ones(ans.shape))
    return gradfun

def make_vjp(func, x):
    '''
    Construct function for vector-Jacobian product
    '''
    end_value = func(*x if isinstance(x, tuple) else x)
    assert end_value.shape == (1, 1) or end_value.shape == () or end_value.shape == (1,)
    def vjp(g): 
        return backward(g, end_value, x)
    return vjp, end_value
     
