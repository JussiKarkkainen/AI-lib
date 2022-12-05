import numpy as np
from AIlib.nn.utils import one_hot
from AIlib.tensor import Tensor
from AIlib.autograd import grad
import AIlib.nn as nn
from AIlib.transform import transform
from AIlib.nn.module import wrap_method
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

def load_dataset():
    transformn = transforms.Compose(
            [transforms.ToTensor()])
    batch_size = 32
    trainset = datasets.MNIST(root='./data', train=True,
                              download=True, transform=transformn)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)
    testset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transformn)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=1)

    x_init = Tensor.zeros(32, 1, 28, 28)
    return train_loader, x_init

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, kernel_size=5, padding=2, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, kernel_size=5, bias=False)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lin1 = nn.Linear(120)
        self.lin2 = nn.Linear(84)
        self.lin3 = nn.Linear(10)
    
    @wrap_method
    def __call__(self, x):
        h1 = self.maxpool1(self.conv1(x)).relu()
        h2 = self.maxpool2(self.conv2(h1)).relu()
        h2 = h2.flatten(start_dim=1)
        h3 = self.lin1(h2).relu()
        h4 = self.lin2(h3).relu()
        out = self.lin3(h4)
        return out

def net_fn(X):
    net = LeNet()
    return net(X)

lossfn = nn.CategoricalCrossEntropyLoss() 

def main():
    network = transform(net_fn)
    optimizer = nn.optim.sgd(1e-3)

    def loss_fn(params, x, y):
        out = network.apply(params, x)
        loss = lossfn(out, y)
        return loss

    def update_weights(params, x, y):
        grads, loss = grad(loss_fn)(params, x, y)
        params, opt_state = optimizer.update(grads, state.opt_state)
        return nn.TrainingState(params, opt_state), loss

    train_loader, x_init = load_dataset()
    init_params = network.init(x_init)
    init_opt_state = optimizer.init(init_params)
    state = nn.TrainingState(params=init_params, opt_state=init_opt_state)

    for epoch in range(10):
        epoch_loss = 0
        for x, y in tqdm(train_loader):
            x = Tensor(np.array(x)).detach()
            y = nn.utils.one_hot(Tensor(np.array(y)), 10).detach()
            state, loss = update_weights(state.params, x, y)
            epoch_loss += loss

        print(f"Loss on epoch: {epoch} was {epoch_loss}")


if __name__ == "__main__":
    main()

