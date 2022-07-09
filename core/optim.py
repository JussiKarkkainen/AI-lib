
class Optim:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for param in self.params:
            param.grad = None
    

class SGD:
    def __init__(self, params, lr):
        self.lr = lr
        slef.params = params

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad  


class RMSprop:



class Adam:
