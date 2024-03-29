import numpy as np
from AIlib.tensor import Tensor
from AIlib.autograd import grad
import AIlib.nn as nn
from AIlib.transform import transform
import matplotlib.pyplot as plt
from AIlib.nn.module import wrap_method
from tqdm import tqdm

Xs = np.linspace(-3., 3., num=256)[:, None]
X = Tensor(Xs)
y = Tensor(np.sin(Xs))
xy = np.sin(Xs)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(256)
        self.l2 = nn.Linear(256)
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
    optimizer = nn.optim.sgd(0.003)
    
    def loss_fn(params, X, y):
        out = network.apply(params, X)
        loss = Tensor.mean((out - y) ** 2)
        return loss

    def update(params, X, y):
        grads, loss = grad(loss_fn)(params, X, y)
        params, opt_state = optimizer.update(grads, state.opt_state)
        return nn.TrainingState(params, opt_state)

    init_params = network.init(X)
    init_opt_state = optimizer.init(init_params)
    state = nn.TrainingState(params=init_params, opt_state=init_opt_state)
    
    for epoch in tqdm(range(1000)):
        state = update(state.params, X, y)
    
    plt.scatter(Xs, xy, label='Data')
    plt.scatter(Xs, np.array(network.apply(state.params, X).data), label="Model prediction")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
