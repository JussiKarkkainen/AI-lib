import ops

def check_node(function, self, y):
    if isinstance(y, Node):
        return function(self, y)
    raise TypeError("Incompatible type") 


class Node:
    def __init__(self):
        pass
    
    def __add__(self, y):
        return check_node(Add, self, y)

    def __mul__(self, y):
        return check_node(Mul, self, y)

    def __div__(self, y):
        return check_node(Div, self, y)

    def __pow__(self, y):
        return check_node(Pow, self, y)

    def __matmul__(self, y):
        return check_node(Matmul, self, y)

class Graph:
    ''' Class for computational graphs
        _graph is a global variable that describes the graph
    '''

    def __init__(self):
        self.ops = set()
    
        global _graph
        _graph = self

