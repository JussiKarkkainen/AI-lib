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
        h1 = Tensor.relu(self.w1(x))
        h2 = Tensor.relu(self.w2(h1))
        out = self.out(h2)
        return out


net = MLP()
optim = SGD(net.parameters, lr=0.01)
lossfn = CrossEntropyLoss 

