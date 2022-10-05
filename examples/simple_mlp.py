import numpy as np
from core.tensor import Tensor
from core.autograd import grad
import core.nn as nn
from core.transform import transform
import core.nn.optim as optim
import matplotlib.pyplot as plt
from core.nn.module import wrap_method

Xs = np.linspace(-2., 2., num=128)[:, None]
X = Tensor(Xs)
y = Tensor(Xs ** 2)
xy = Xs ** 2

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(128)
        self.l2 = nn.Linear(128)
        self.l3 = nn.Linear(1)
    
    @wrap_method
    def __call__(self, x):
        out = self.l1(x).sigmoid()
        out = self.l2(out).sigmoid()
        out = self.l3(out)
        return out

def net_fn(x):
    net = Net()
    return net(x)

def main():
    network = transform(net_fn)
    optimizer = optim.sgd(0.003)
    
    def loss(params, X, y):
        out = network.apply(params, X)
        loss = Tensor.mean((out - y) ** 2)
        return loss

    def update(params, X, y):
        grads = grad(loss)(params, X, y)
        params, opt_state = optimizer.update(grads, state.opt_state)
        return nn.TrainingState(params, opt_state)

    def evaluate(params, X, y):
        out = network.apply(params, X)
        predictions = np.argmax(out, axis=-1)
        return Tensor.mean(predictions == y)
    
    init_params = network.init(X)
    init_opt_state = optimizer.init(init_params)
    state = nn.TrainingState(params=init_params, opt_state=init_opt_state)
    
    for epoch in range(1000):
        state = update(state.params, X, y)
    
    plt.scatter(Xs, xy, label='Data')
    plt.scatter(Xs, np.array(network.apply(state.params, X).data), label="Model prediction")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()