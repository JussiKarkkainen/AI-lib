import os
import sys
import numpy as np
import gzip
from core.tensor import Tensor

def load_mnist():
    train_img = gzip.open("../dataset/mnist/train-images-idx3-ubyte.gz").read()
    train_img = Tensor(np.frombuffer(train_img, dtype=np.uint8)[0x10:].reshape((-1, 28, 28)).astype(np.float32))

    train_labels = gzip.open("../dataset/mnist/train-labels-idx1-ubyte.gz").read()
    train_labels = Tensor(np.frombuffer(train_labels, dtype=np.uint8)[8:])
    
    test_img = gzip.open("../dataset/mnist/t10k-images-idx3-ubyte.gz").read()
    test_img = Tensor(np.frombuffer(test_img, dtype=np.uint8)[0x10:].reshape((-1, 28, 28)).astype(np.float32))

    test_labels = gzip.open("../dataset/mnist/t10k-labels-idx1-ubyte.gz").read()
    test_labels = Tensor(np.frombuffer(test_labels, dtype=np.uint8)[8:])

    return train_img, train_labels, test_img, test_labels

def load_tiny_shakespeare():
    text = open('input.txt', 'r').read()
    train_dataset = Dataset(text)
    return train_dataset
