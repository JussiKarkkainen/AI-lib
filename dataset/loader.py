import os
import sys
import numpy as np
import gzip

def load_mnist():
    X_train = gzip.open("../dataset/mnist/train-images-idx3-ubyte.gz").read()
    X_train = np.frombuffer(X_train, dtype=np.uint8)[0x10:].reshape((-1, 28*28)).astype(np.float32) 

    Y_train = gzip.open("../dataset/mnist/train-labels-idx1-ubyte.gz").read()
    Y_train = np.frombuffer(Y_train, dtype=np.uint8)[8:]
    
    X_test = gzip.open("../dataset/mnist/t10k-images-idx3-ubyte.gz").read()
    X_test = np.frombuffer(X_train, dtype=np.uint8).reshape((-1, 28*28)).astype(np.float32) 

    Y_test = gzip.open("../dataset/mnist/t10k-labels-idx1-ubyte.gz").read()
    Y_test = np.frombuffer(Y_train, dtype=np.uint8)

    return X_train, X_test, Y_train, Y_test
