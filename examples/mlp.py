#!/usr/bin/env python3

from core.tensor import Tensor
from core.optim import SGD
from dataset.loader import load_mnist
from utils.layer import Linear
from models import Model

train_img, train_labels, test_img, test_labels = load_mnist()


class MLP(Model):

