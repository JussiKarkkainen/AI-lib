import numpy as np
from core.nn.utils import one_hot
from core.tensor import Tensor
from core.autograd import grad
import core.nn as nn
from core.transform import transform
import core.nn.optim
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from core.nn.module import wrap_method


def load_dataset():
    transformn = transforms.Compose(
        [transforms.ToTensor()])
    batch_size = 256
    trainset = datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transformn)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transformn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    train_loader = iter(trainloader)
    x_init, y_init = next(train_loader)
    x_init, y_init = Tensor(np.array(x_init)), Tensor(np.array(y_init))
    return train_loader, x_init, y_init 

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(256)
        self.w2 = nn.Linear(256)
        self.out = nn.Linear(10)
    
    @wrap_method
    def __call__(self, x):
        x = x.flatten()
        h1 = self.w1(x).relu()
        h2 = self.w2(h1).relu()
        out = self.out(h2)
        return out

def net_fn(x):
    net = MLP()
    return net(x)

lossfn = nn.CrossEntropyLoss()

def main():
    network = transform(net_fn)
    optimizer = optim.sgd

    def loss(params, X, y):
        out = network.apply(params, X)
        out = lossfn(X, y)
    
    def update(state, X, y):
        grads = grad(loss)(state.params, X, y)
        updates, opt_state = optimizer.update(grads, state.opt_state)
        params = optim.apply_updates(state.params, updates)
        return nn.TrainingState(params, opt_state)
   
    def evaluate(params, X, y):
        out = network.apply(params, X, y)
        predictions = jnp.argmax(out, axis=-1)
        return Tensor.mean(predictions == y)

    train_loader, x_init, y_init = load_dataset()
    init_params = network.init(x_init, y_init)
    init_opt_state = optimizer.init(initial_params)

    state = nn.TrainingState(init_params, init_opt_state)

    for epoch in range(10):
        for X, y in train_loader:
            X = Tensor(np.array(X))
            y = Tensor(np.array(y))
            state = update(state, X, y)

        accuracy = evaluate(state.params, X, y)
        print(f"epoch: {epoch}, accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()
