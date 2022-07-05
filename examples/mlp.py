#!/usr/bin/env python3

import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.loader import load_mnist


def mlp(X):
    X = X.reshape(-1, 784)
    H1 = F.relu(torch.matmul(X, params[0]) + params[1])
    H2 = F.relu(torch.matmul(H1, params[2]) + params[3])
    return F.softmax(F.relu(H2, params[4] + params[5]))

def init_params():
    W1 = torch.normal(0, 0.01, (784, 256), requires_grad=True)
    b1 = torch.zeros(256, requires_grad=True)
    W2 = torch.normal(0, 0.01, (256, 256), requires_grad=True)
    b2 = torch.zeros(256, requires_grad=True)
    W3 = torch.normal(0, 0.01, (256, 10), requires_grad=True)
    b3 = torch.zeros(10, requires_grad=True)

    params = [W1, b1, W2, b2, W3, b3]
    return params
    
if __name__ == "__main__":
    params = init_params()
    lr = 0.1
    optim = torch.optim.SGD(params, lr)
    loss = nn.CrossEntropyLoss()
    train_iter, train_labels, test_iter, test_labels = load_mnist()

    losses = []

    for epoch in range(10):
        for X, y in zip(train_iter, train_labels):
            l = loss(mlp(torch.from_numpy(X)), y)
            l.backward()
            optim.zero_grad()
            optim.step()
            losses.append(l)
            
        print(losses)


