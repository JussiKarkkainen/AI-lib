# Based on https://github.com/mattjj/autodidact
import numpy as np



class Node:
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        self.parents = parents
        self.recipe = (fun, value, args, kwargs, parent_argnums)



def topological_sort(end_node):
    pass

def backward(graph, end_node):
    pass




def grad(func, argnum=0):
    '''
    Constuct the gradient function that returns the gradient of 
    the given function w.r.t x
    '''
    def gradfunc(*args, *kwargs):
        fun = lambda x: func(*subval(args, argnum, x), **kwargs)
        vjp, ans = make_vjp(fun, args[argnum])
        
        return vjp(np.ones_like(ans))
    return gradfunc















