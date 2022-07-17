

class Graph:
    ''' Class for computational graphs
        _graph is a global variable that describes the graph
    '''

    def __init__(self):
        self.ops = set()
    
        global _graph
        _graph = self
