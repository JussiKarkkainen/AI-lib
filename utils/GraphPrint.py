from graphviz import Digraph
from core.buffer import Ops

def trace(root):
    nodes, edges = list(), list()
    def graph(root):
        if root not in nodes:
            nodes.append(root)
            if root._graph is not None:
                for child in [root._graph]:
                    edges.append(child)
                    [graph(n) for n in root._graph.parents]
    graph(root)
    return nodes, edges

def print_graph(root, format='svg', rankdir='LR'):
    ''' Visualizing graphs for backpropagation'''
    viz = Digraph(format=format, graph_attr={'rankdir': rankdir})
    viz.attr(rankdir='LR', size='10, 8')
    nodes, edges = trace(root)
    for n in nodes:
        viz.node(name=str(id(n)), label="data: %s" % str(n.data), shape='record')
    for i, n in enumerate(nodes):
        if n._graph is not None:
            for p in n._graph.parents:
                viz.edge(str(id(p)), str(id(n)), label=(str(edges[i]).split("object")[0]))
    return viz
