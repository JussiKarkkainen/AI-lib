class Optim:
    def __init__(self, params):
        self.params = params
    
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.grad = None

class SGD(Optim):
    def __init__(self, params, lr, momentum=0, weight_decay=0.0):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.velocity[]
        for param = params:
            self.velocity.append(np.zeros_like(param.grad)

    def step(self):
        for param, v in zip(self.params, self.velocity):
            v = self.momentum * self.velocity + param.grad + self.weight_decay * param.grad
            param -= self.lr * v

class RMSprop(Optim):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8) 
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = [Tensor.zeros(param.shape, param[0].device, requires_grad=False) for param in self.params]

    def step(self):
        for param in self.params:
            self.v = self.alpha * self.v + (1 - self.alpha) * param.grad.pow(2)
            param -= (self.lr * param.grad).div(self.v.pow(0.5) + self.eps)

class Adam(Optim):
    def __init__(self, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, maximize=False):
        super().__init__()
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.maximize = maximize

    def step(self):
        pass

