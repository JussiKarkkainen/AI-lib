from tensor import Tensor

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
