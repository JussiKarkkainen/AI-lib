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

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(256)
        self.w2 = nn.Linear(256)
        self.out = nn.Linear(10)

    def forward(self, x):
        x = x.flatten()
        h1 = self.w1(x).relu()
        h2 = self.w2(h1).relu()
        out = self.out(h2)
        return out

lossfn = nn.CrossEntropyLoss()

def loss(X, y):
    net = MLP()
    return lossfn(net(X), y)

loss_fn_t = transform(loss)
x_init, y_init = next(train_loader)
x_init, y_init = Tensor(np.array(x_init)), Tensor(np.array(y_init))
params = loss_fn_t.init(x_init, y_init)
print(params)
'''
optim = SGD(params, lr=0.01)

print("starting training")
for epoch in range(10):
    for X, y in train_loader:
        X = Tensor(np.array(X))
        y = Tensor(np.array(y))
        grads = grad(loss_fn_t.apply)(params, X, y)
        params = optim.step(params, grads)

    print("training")

'''
