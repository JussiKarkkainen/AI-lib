from core.tensor import Tensor
from nn.module import Module
from nn.layer import Conv2d, Linear
from dataset.loader import load_mnist
from core.optim import SGD
from nn.loss import CrossEntropyLoss

train_img, train_labels, test_img, test_labels = load_mnist()

class AlexNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 6, kernel_size=5, padding=2)
        self.avgpool1 = AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(6, 16, kernel_size=5)
        self.avgpool2 = AvgPool(kernel_size=2, stride=2)
        self.lin1 = Linear(400, 120)
        self.lin2 = Linear(120, 84)
        self.lin3 = Linear(84, 10)

    def forward(self, x):
        h1 = self.avgpool1(self.conv1(x)).sigmoid()
        h2 = self.avgpool2(self.conv2(h1)).sigmoid()
        h2 = h2.flatten()
        h3 = self.lin1(h2).sigmoid()
        h4 = self.lin2(h3).sigmoid()
        out = self.lin3(h4)
        return out

net = AlexNet()
optim = SGD(net.parameters(), lr=0.01)
lossfn = CrossEntropyLoss()
num_epochs = 10

for epoch in range(num_epochs):
    for X, y in zip(train_img, train_labels):
        y_hat = net(X)
        grads = grad(loss)(net.parameters(), X, y)
        # Study jax training loops, my optimizer won't work with this



