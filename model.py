from tensor import Tensor
from ops import Function
import numpy as np

class Linear(Function):
    def __init__(self, in_nodes, out_nodes):
        self.weights = Tensor.randn((in_nodes, out_nodes))
        self.bias = Tensor.zeros((out_nodes))
        self.type = 'Linear'

    def forward(self, x):
        self.input = x
        return np.dot(self.input, self.weights.data) + self.bias
        
    def backward(self, d_y):
        self.weights.grad += np.dot(self.input.T, d_y)
        self.bias.grad += np.sum(d_y, axis=0, keepdim=True)
        grad_input = np.dot(d_y, self.weights.data.T)
        return grad_input

    def get_params(self):
        return [self.weights, self.bias]



class Model:
    ''' Class for making a model '''
    
    def __init__(self):
        self.computation_graph = []
        self.parameters = []

    def add(self, layer):
        self.computation_graph.append(layer)
        self.parameters += layer.get_params()

    def initalize_net(self):
        for l in self.computation_graph:
            weights, bias = l.get_params()
            weights.data = Tensor.randn(weights.data.shape)
            bias.data = 0.


    def train(self, data, target, batch_size, num_epochs, optim, loss_fn):
        


    def predict(self, data):
