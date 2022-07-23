
class Layer:
    def __init__(self):
        self.parameters = []

    def get_params(self):
        return self.parameters

class Linear(Layer):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.weights = Tensor.randn((num_inputs, num_outputs), requires_grad=True)
        self.bias = Tensor.zeros((1, num_outputs), requires_grad=True)

        self.parameters.append(self.weights)
        self.parameters.append(self.bias)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias
