import numpy as np
from AIlib.nn.utils import one_hot
from AIlib.tensor import Tensor
from AIlib.autograd import grad
import AIlib.nn as nn
from AIlib.transform import transform
import AIlib.nn.optim as optim
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from AIlib.nn.module import wrap_method
from tqdm import tqdm

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
    
    x_init, y_init = Tensor.zeros((256, 784)), Tensor.zeros((256,))
    return trainloader, x_init, y_init 

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(256)
        self.w2 = nn.Linear(256)
        self.out = nn.Linear(10)
    
    @wrap_method
    def __call__(self, x):
        out = self.w1(x).tanh()
        out = self.w2(out).tanh()
        out = self.out(out)
        return out

def net_fn(x):
    net = MLP()
    return net(x)

lossfn = nn.CrossEntropyLoss()

def main():
    network = transform(net_fn)
    optimizer = optim.sgd(0.001)
    
    def loss_fn(params, X, y):
        out = network.apply(params, X)
        loss = lossfn(out, y)
        return loss

    def update(params, X, y):
        grads, loss = grad(loss_fn)(params, X, y)
        params, opt_state = optimizer.update(grads, state.opt_state)
        return nn.TrainingState(params, opt_state), loss
   
    train_loader, x_init, y_init = load_dataset()
    init_params = network.init(x_init.flatten())
    init_opt_state = optimizer.init(init_params)
    state = nn.TrainingState(params=init_params, opt_state=init_opt_state)
    
    print("Starting training")
    for epoch in tqdm(range(5)):
        epoch_loss = 0
        for X, y in train_loader:
            X = Tensor(np.array(X)).flatten().detach()
            y = np.array(y)
            state, loss = update(state.params, X, y)
            epoch_loss += loss
        
        print(f"loss on epoch: {epoch} was {epoch_loss}")

if __name__ == "__main__":
    main()
