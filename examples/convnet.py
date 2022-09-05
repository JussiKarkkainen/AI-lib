from core.tensor import Tensor
from core.autograd import grad
from nn.module import Module
import nn.layer as nn
from dataset.loader import load_mnist
from core.optim import SGD
from nn.loss import CrossEntropyLoss
import matplotlib.pyplot as plt

train_img, train_labels, test_img, test_labels = load_mnist()

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
        h1 = self.maxpool1(self.conv1(x)).sigmoid()
        h2 = self.maxpool2(self.conv2(h1)).sigmoid()
        h2 = h2.flatten()
        h3 = self.lin1(h2).sigmoid()
        h4 = self.lin2(h3).sigmoid()
        out = self.lin3(h4)
        return out

net = ConvNet()
params = net.parameters()
optim = SGD(params, lr=0.01)
lossfn = CrossEntropyLoss
num_epochs = 10

for epoch in range(num_epochs):
    for X, y in zip(train_img, train_labels):
        y_hat = net(X)
        grads = grad(lossfn, 0)(params, X, y)
        params = optim.step(params, grads) 

    loss = lossfn(net(X), y)
    print(f"loss on epoch: {epoch} is {loss}")
