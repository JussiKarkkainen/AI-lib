from core.tensor import Tensor

class Optim:
    def __init__(self, params):
        self.params = params
    
    def step(self):
        raise NotImplementedError

class SGD(Optim):
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        

    def step(self, params, grads):
        velocity = []
        for grad in grads:
            velocity.append(Tensor.zeros(grad.shape))
        for param, v, grad in zip(params, self.velocity, grads):
            v = self.momentum * self.velocity + self.weight_decay * grad
            param -= self.lr * v
        return params

class RMSprop(Optim):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8): 
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = [Tensor.zeros(param.shape, param[0].device, requires_grad=False) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * param.grad.pow(2)
            param -= (self.lr * param.grad).div(self.v[i].pow(0.5) + self.eps)

class Adam(Optim):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
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


