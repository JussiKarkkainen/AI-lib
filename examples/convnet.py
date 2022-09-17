import numpy as np
from core.nn.utils import one_hot
from core.tensor import Tensor
from core.autograd import grad
from core.nn.module import Module
import core.nn.layer as nn
from dataset.loader import load_mnist
from core.optim import SGD
from core.nn.loss import CrossEntropyLoss
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 256

trainset = datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

class ConvNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lin1 = nn.Linear(400, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)

    def forward(self, x):
        out = params 
        h1 = self.maxpool1(self.conv1(x)).relu()
        h2 = self.maxpool2(self.conv2(h1)).relu()
        h2 = h2.flatten(start_dim=1)
        h3 = self.lin1(h2).relu()
        h4 = self.lin2(h3).relu()
        out = self.lin3(h4)
        return out

net = ConvNet()
params = net.parameters()
optim = SGD(params, lr=0.01)
lossfn = CrossEntropyLoss() 
num_epochs = 10

def loss(params, X, y):
    y_hat = net(X)
    out = lossfn(y_hat, y)
    return out

for epoch in range(num_epochs):
    for X, y in trainloader:
        X = Tensor(np.array(X))
        y = Tensor(np.array(y))
        grads = grad(loss, 0)(params, X, y)
        params = optim.step(params, grads) 

    loss = lossfn(net(X), y)
    print(f"loss on epoch: {epoch} is {loss}")
