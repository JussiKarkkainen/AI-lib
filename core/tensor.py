import numpy as np

class Tensor:
    def __init__(self, data, creators=None, creator_op=None, device=None, requires_grad=False,
                 ident=None):
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.device = device
        self.creators = creators
        self.creator_op = creator_op
        self.children= {}

        # There should be a better way to do this
        if ident == None:
            ident = np.random.randint(0, 100000)
        self.ident = ident

        if creators is not None:
            for creator in creators:
                if self.ident not in creator.children:
                    creator.children[self.ident] = 1
                else:
                    creator.children[self.ident] += 1


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

    def all_children_accounted_for(self):
        for ident, cnt in self.children.items():
            if cnt != 0:
                return False
            else:
                return True
    
    '''
    This is a temporary implementation, a more refined one will come later
    '''
    def backward(self, grad=None, grad_origin=None):
        if self.requires_grad:
            if grad_origin is not None:
                if self.children[grad_origin.ident] == 0:
                    raise Exception("Cannot backpropagate more than once")
                else:
                    self.children[grad_origin.ident] -= 1
            if self.grad == None:
                self.grad = grad
            else:
                self.grad += grad
            if self.creators is not None and (self.all_children_accounted_for() or 
                    grad_origin is None):

                if self.creator_op == "add":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                if self.creato_op == "neg":
                    self.creators[0].backward(self.grad.__neg__())

    def __add__(self, other):
        if (self.requires_grad and other.requires_grad):
            return Tensor(self.data + other.data, creators=[self, other], creator_op="add",
                          requires_grad=True)

        return Tensor(self.data + other.data)
    
    def __neg__(self):
        if self.requires_grad:
            return Tensor(self.data * -1, creators=[self], creator_op="neg",
                          requires_grad=True)

        return Tensor(self.data * -1)

    def __sub__(self, other):
        if (self.requires_grad and other.requires_grad):
            return Tensor(self.data - other.data, creators=[self, other], creator_op="sub",
                          requires_grad=True)

        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if (self.requires_grad and other.requires_grad):
            return Tensor(self.data * other.data, creators=[self, other], creator_op="mul",
                          requires_grad=True)

        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.requires_grad:
            return Tensor(self.data.sum(dim), creators=[self], creator_op="sum",
                          requires_grad=True)

        return Tensor(self.data.sum(dim))

    def transpose(self):
        if self.requires_grad:
            return Tensor(self.data.transpose(), creators=[self], creator_op="transpose",
                          requires_grad=True)

        return Tensor(self.data.transpose())
    
    def expand(self, dim, copies):
        trans_cmd = list(range(0,, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if self.requires_grad:
            return Tensor(new_data, creators=[self], creator_op="expand " + str(dim),
                          requires_grad=True)

        return Tensor(new_data)

    '''    
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
        
        Calculate the backward pass:
        graph: topologically ordered array of graph nodes. The gradient of the final
               node is set to 1.
        Function returns gradients of nodes in the same order as the input arg
        

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

    '''
    
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


