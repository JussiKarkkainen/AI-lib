import numpy as np
from ops import _graph


class Tensor:
    def __init__(self, data, device=None, requires_grad=True):
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        else:
            self.data = data
        self.grad = 0
        self.requires_grad = requires_grad
        self.device = device

    def __repr__(self):
        return f"<Tensor: data={self.data} Grad={self.grad}>"
        
    @property
    def dtype(self):
        return np.float32

    @property
    def shape(self):
        return np.shape(self.data)

    def device(self):
        return self.device
    
    def topological_sort(end_node=None, graph=_graph):
        order = []
        visited_nodes = set()
        def _topo(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                if isinstance(node, Ops):
                    for input_node in node.inputs:
                        _topo(input_node)
                ordering.append(node)
        
        if head_node == None:
            for node in graph.ops:
                _topo(node)
        else:
            _topo(head_node)

        return order


    def backward(graph, end_node=None):
        '''
        Calculate the backward pass:
        graph: topologically ordered array of graph nodes. The gradient of the final
               node is set to 1.
        Function returns gradients of nodes in the same order as the input arg
        '''

        graph[-1].grad = 1
        visited = set()
        for node in reversed(toposort(end_node)):
            if isinstance(node, Ops):
                inputs = node.inputs
                grads = node.backward(*[x.value for x in inputs], dout=node.gradient)
                for ins, grad in zip(inputs, grads):
                    if ins not in visited:
                        ins.gradient = grad
                    else:
                        ins.gradient += grad
                    visited.add(ins)
        return [node.gradient for node in order]


    def __getitem__(self, key):
        return key


    # Functions for creating tensors #

    @classmethod 
    def arange(cls, end, start=0, **kwargs):
        return cls(np.arange(start, end).astype(np.float32), **kwargs)
    
    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)
    
    @classmethod
    def ones(cls, *shape, **kwargs):
        return cls(np.ones(shape, dtype=np.float32), **kwargs) 
    
    @classmethod
    def rand(cls, *shape, **kwargs):
        return cls(np.random.rand(*shape).astype(np.float32), **kwargs)

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(np.random.randn(*shape).astype(np.float32), **kwargs)
    
    @classmethod
    def zeros(cls, *shape, **kwargs):
        return cls(np.zeros(shape, dtype=np.float32), **kwargs)


