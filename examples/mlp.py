from core.tensor import Tensor
from core.optim import SGD
from dataset.loader import load_mnist
from utils.layer import Linear
from utils.loss import CrossEntropyLoss
from models import Model

train_img, train_labels, test_img, test_labels = load_mnist()

class MLP(Model):
    def __init__(self):
        super().__init__()
        self.w1 = Linear(784, 256)
        self.w2 = Linear(256, 256)
        self.out = Linear(256, 10)

    def forward(self, x):
        h1 = (self.w1(x)).relu()
        h2 = (self.w2(h1)).relu()
        out = self.out(h2)
        return out


net = MLP()
params = net.parameters()
optim = SGD()
lossfn = CrossEntropyLoss()

for epoch in range(10):
    for X, y in zip(train_img, train_labels):
        y_hat = net(X)
        loss = loddfn(y_hat, y)
        grads = grad(lossfn, 0)(params, X, y)
        params = optim(params, lr=0.01)


