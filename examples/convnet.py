import numpy as np
from core.nn.utils import one_hot
from core.tensor import Tensor
from core.autograd import grad
import core.nn as nn
from core.transform import transform
from core.nn.optim import SGD
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

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

class ConvNet(nn.Module):
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
        h1 = self.maxpool1(self.conv1(x)).relu()
        h2 = self.maxpool2(self.conv2(h1)).relu()
        h2 = h2.flatten(start_dim=1)
        h3 = self.lin1(h2).relu()
        h4 = self.lin2(h3).relu()
        out = self.lin3(h4)
        return out

lossfn = nn.CrossEntropyLoss() 
num_epochs = 10

def loss(X, y):
    net = ConvNet()
    return lossfn(net(X), y)

loss_fn_t = transform(loss)
x_init, y_init = next(train_loader)
x_init, y_init = Tensor(np.array(x_init)), Tensor(np.array(y_init))
params = loss_fn_t.init(x_init, y_init)
optim = SGD(params, lr=0.01)

print("starting training")
for epoch in range(num_epochs):
    for X, y in trainloader:
        X = Tensor(np.array(X))
        y = Tensor(np.array(y))
        grads = grad(loss_fn_t.apply)(params, X, y)
        params = optim.step(params, grads) 

    # Shoudn't use two forward passes, way too slow
    loss = lossfn(net(X), y)
    print(f"loss on epoch: {epoch} is {loss.data}")
