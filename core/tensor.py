import numpy as np

class Tensor:
    def __init__(self, data, device=None, requires_grad=True):
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.device = device

        # Used for storing computational graph
        self._graph = None


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

    
    def topological_sort(node=_graph):
        order = []
        visited_nodes = set()
        def _topo(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                if node._graph: 
                    for input_node in node.inputs:
                        _topo(input_node)
                order.append(node)
        
            return order


    def backward(graph, end_node=None):
        
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

    # Functions for creating tensors 
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


