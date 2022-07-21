from graphviz import Digraph
from core.ops import Ops

def print_graph(graph):
    ''' Visualizing graphs for backpropagation'''
    viz = DiGraph()
    viz.attr(rankdir='LR', size='10, 8')
    viz.attr('node', shape='circle')
    for node in graph:
        viz.node(node.name, label=node.name.spilt('/')[0], shape='circle')
    for node in graph:
        if isinstance(node, Ops):
            for i in node.inputs:
                viz.edge(i.name, node.name, label=i.name)

    return viz
