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
        
        self.velocity = []
        for param in params:
            self.velocity.append(np.zeros_like(param.grad))

    def step(self):
        for param, v in zip(self.params, self.velocity):
            v = self.momentum * self.velocity + param.grad + self.weight_decay * param.grad
            param -= self.lr * v

class RMSprop(Optim):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8): 
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = [Tensor.zeros(param.shape, param[0].device, requires_grad=False) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * param.grad.pow(2)
            param -= (self.lr * param.grad).div(self.v[i].pow(0.5) + self.eps)

class Adam(Optim):
    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        
        self.m = [Tensor.zeros(param.shape, param[0].device, requires_grad=False) for param in self.params]
        self.v = [Tensor.zeros(param.shape, param[0].device, requires_grad=False) for param in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (param.grad * param_grad)
            param -= (self.lr * (self.m[i].div(1 - self.betas[0].pow(self.t)))).div(((((((self.v[i].div(1 - self.betas[1].pow(self.t))) + self.eps)) + self.eps))))


